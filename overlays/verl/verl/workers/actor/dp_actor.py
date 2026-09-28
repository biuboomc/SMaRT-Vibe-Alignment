# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from collections import OrderedDict

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.luffy_core_algos import compute_sft_pure_loss, compute_token_on_off_policy_loss
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from verl.workers.actor.ephemeral_lora import (
    apply_ephemeral_lora_to_model,
    extract_ephemeral_lora_paths_from_micro_batch,
    slice_micro_batch_rows,
)

__all__ = [
    "DataParallelPPOActor",
    "compute_sft_variant_losses",
    "compute_sft_category_weighted_loss",
    "resolve_sft_category_weights",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SFT_LORA_VARIANTS = ("knowledge", "no_op", "random")
SFT_CATEGORIES = ("knowledge", "behavior", "no_op", "random")


def _coerce_variant_values(variant_values, batch_size: int) -> list[str | None]:
    if variant_values is None:
        return [None] * batch_size
    if hasattr(variant_values, "tolist"):
        variant_values = variant_values.tolist()
    elif not isinstance(variant_values, (list, tuple)):
        variant_values = [variant_values]
    values = list(variant_values)
    if len(values) < batch_size:
        values.extend([None] * (batch_size - len(values)))
    return [None if value in (None, "", "None") else str(value) for value in values[:batch_size]]


def _select_lora_variant_values(model_inputs: dict, batch_size: int) -> list[str | None]:
    for key in ("ephemeral_lora_variant", "lora_variant", "luffy_teacher_lora_variant"):
        values = model_inputs.get(key)
        if values is not None:
            coerced = _coerce_variant_values(values, batch_size)
            if any(value is not None for value in coerced):
                return coerced

    extra_infos = model_inputs.get("extra_info")
    if extra_infos is None:
        return [None] * batch_size
    if hasattr(extra_infos, "tolist"):
        extra_infos = extra_infos.tolist()
    values = []
    for item in list(extra_infos)[:batch_size]:
        if isinstance(item, dict):
            values.append(item.get("ephemeral_lora_variant") or item.get("lora_variant"))
        else:
            values.append(None)
    return _coerce_variant_values(values, batch_size)


def _select_sft_category_values(model_inputs: dict, batch_size: int) -> list[str | None]:
    sample_types = _coerce_variant_values(model_inputs.get("luffy_teacher_sample_type"), batch_size)
    update_types = _coerce_variant_values(model_inputs.get("update_type"), batch_size)
    categories: list[str | None] = []
    for sample_type, update_type in zip(sample_types, update_types):
        if sample_type == "no_change":
            categories.append("no_op")
        elif sample_type == "random_change":
            categories.append("random")
        elif sample_type == "changed" and update_type in {"knowledge", "behavior"}:
            categories.append(update_type)
        else:
            categories.append(None)
    return categories


def _linear_ramp(*, step: int, start_value: float, end_value: float, start_step: int, ramp_steps: int) -> float:
    if ramp_steps <= 1:
        return float(end_value if step >= start_step else start_value)
    end_step = start_step + ramp_steps - 1
    if step <= start_step:
        return float(start_value)
    if step >= end_step:
        return float(end_value)
    progress = float(step - start_step) / float(end_step - start_step)
    return float(start_value + progress * (end_value - start_value))


def resolve_sft_category_weights(config: ActorConfig, global_step: int) -> dict[str, float]:
    ramp_start_step = int(config.get("sft_negative_loss_ramp_start_step", 1))
    ramp_steps = int(config.get("sft_negative_loss_ramp_steps", 1))
    return {
        "knowledge": 1.0,
        "behavior": 1.0,
        "no_op": _linear_ramp(
            step=global_step,
            start_value=float(config.get("sft_no_op_loss_weight_start", 0.0)),
            end_value=float(config.get("sft_no_op_loss_weight_end", 1.0)),
            start_step=ramp_start_step,
            ramp_steps=ramp_steps,
        ),
        "random": _linear_ramp(
            step=global_step,
            start_value=float(config.get("sft_random_loss_weight_start", 0.0)),
            end_value=float(config.get("sft_random_loss_weight_end", 1.0)),
            start_step=ramp_start_step,
            ramp_steps=ramp_steps,
        ),
    }


def compute_sft_category_weighted_loss(
    *,
    log_prob: torch.Tensor,
    eos_mask: torch.Tensor,
    prefix_mask: torch.Tensor,
    category_values,
    category_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = int(log_prob.shape[0])
    categories = _coerce_variant_values(category_values, batch_size)
    teacher_token_mask = prefix_mask.to(dtype=torch.bool) & eos_mask.to(dtype=torch.bool)
    nll = -log_prob
    weighted_nll_sum = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
    all_teacher_token_count = teacher_token_mask.sum().to(dtype=log_prob.dtype)
    metrics: dict[str, float] = {}
    for category in SFT_CATEGORIES:
        row_mask = torch.as_tensor(
            [categories[index] == category for index in range(batch_size)],
            device=log_prob.device,
            dtype=torch.bool,
        )
        token_mask = teacher_token_mask & row_mask.unsqueeze(-1)
        token_count = token_mask.sum()
        nll_sum = (nll * token_mask.to(dtype=nll.dtype)).sum()
        weight = float(category_weights.get(category, 1.0))
        weighted_nll_sum = weighted_nll_sum + weight * nll_sum
        denominator = token_count.to(dtype=nll.dtype).clamp_min(1.0)
        category_loss = nll_sum / denominator
        metrics[f"actor/sft_category_loss/{category}"] = float(category_loss.detach().item())
        metrics[f"actor/sft_category_rows/{category}"] = float(row_mask.sum().detach().item())
        metrics[f"actor/sft_category_nll_sum/{category}"] = float(nll_sum.detach().item())
        metrics[f"actor/sft_category_token_count/{category}"] = float(token_count.detach().item())
        metrics[f"actor/sft_category_weight/{category}"] = weight
        metrics[f"actor/sft_category_weighted_nll_sum/{category}"] = float((weight * nll_sum).detach().item())
    weighted_loss = weighted_nll_sum / all_teacher_token_count.clamp_min(1.0)
    metrics["actor/sft_category_weighted_loss"] = float(weighted_loss.detach().item())
    metrics["actor/sft_category_all_teacher_tokens"] = float(all_teacher_token_count.detach().item())
    return weighted_loss, metrics


def compute_sft_variant_losses(
    *,
    log_prob: torch.Tensor,
    eos_mask: torch.Tensor,
    prefix_mask: torch.Tensor,
    variant_values,
) -> dict[str, float]:
    batch_size = int(log_prob.shape[0])
    variants = _coerce_variant_values(variant_values, batch_size)
    teacher_row_mask = prefix_mask.to(dtype=torch.bool).any(dim=-1)
    metrics: dict[str, float] = {}
    for variant_name in SFT_LORA_VARIANTS:
        row_mask_values = [
            bool(teacher_row_mask[index].detach().item()) and variants[index] == variant_name
            for index in range(batch_size)
        ]
        row_mask = torch.as_tensor(row_mask_values, device=log_prob.device, dtype=torch.bool)
        metric_prefix = f"actor/sft_multitask_loss/{variant_name}"
        rows_prefix = f"actor/sft_multitask_rows/{variant_name}"
        metrics[metric_prefix] = 0.0
        metrics[rows_prefix] = float(row_mask.sum().detach().item())
        if row_mask.any():
            variant_loss = compute_sft_pure_loss(log_prob=log_prob[row_mask], eos_mask=eos_mask[row_mask])
            metrics[metric_prefix] = float(variant_loss.detach().item())
    return metrics


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        self.ephemeral_lora_loss_path_key = str(self.config.get("ephemeral_lora_loss_path_key", "ephemeral_lora_path"))
        self.ephemeral_lora_loss_cache_size = int(self.config.get("ephemeral_lora_loss_cache_size", 1))
        self.ephemeral_lora_loss_require_homogeneous_micro_batch = bool(
            self.config.get("ephemeral_lora_loss_require_homogeneous_micro_batch", True)
        )
        self._ephemeral_lora_loss_cache = OrderedDict()
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        batch_size = int(micro_batch["responses"].shape[0])
        lora_paths = extract_ephemeral_lora_paths_from_micro_batch(
            micro_batch,
            batch_size=batch_size,
            path_key=self.ephemeral_lora_loss_path_key,
        )
        active_paths = [path for path in lora_paths if path is not None]
        if not active_paths:
            return self._forward_micro_batch_without_ephemeral_lora(
                micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )

        unique_paths = list(dict.fromkeys(lora_paths))
        if len(unique_paths) > 1 and self.ephemeral_lora_loss_require_homogeneous_micro_batch:
            preview_paths = [str(path) if path is not None else "<none>" for path in unique_paths[:4]]
            raise ValueError(
                "ephemeral LoRA loss requires each actor microbatch to use one LoRA path; "
                f"got {len(unique_paths)} paths in a microbatch: {preview_paths}. "
                "Set actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1, or implement a "
                "row-wise multi-LoRA loss path before increasing it. This avoids FSDP/NCCL "
                "collective mismatches from multiple hooked forwards in one microbatch."
            )
        if len(unique_paths) == 1:
            with apply_ephemeral_lora_to_model(
                self.actor_module,
                unique_paths[0],
                cache=self._ephemeral_lora_loss_cache,
                max_cache_entries=self.ephemeral_lora_loss_cache_size,
            ):
                return self._forward_micro_batch_without_ephemeral_lora(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )

        grouped_indices: dict[str | None, list[int]] = {}
        for row_index, lora_path in enumerate(lora_paths):
            grouped_indices.setdefault(lora_path, []).append(row_index)

        grouped_outputs = []
        for lora_path, indices in grouped_indices.items():
            sub_micro_batch = slice_micro_batch_rows(micro_batch, indices, batch_size=batch_size)
            with apply_ephemeral_lora_to_model(
                self.actor_module,
                lora_path,
                cache=self._ephemeral_lora_loss_cache,
                max_cache_entries=self.ephemeral_lora_loss_cache_size,
            ):
                sub_output = self._forward_micro_batch_without_ephemeral_lora(
                    sub_micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            grouped_outputs.append((indices, sub_output))

        merged_outputs: dict[str, torch.Tensor] = {}
        for output_key in grouped_outputs[0][1].keys():
            row_outputs: list[torch.Tensor | None] = [None] * batch_size
            for indices, sub_output in grouped_outputs:
                for sub_row_index, original_row_index in enumerate(indices):
                    row_outputs[original_row_index] = sub_output[output_key][sub_row_index : sub_row_index + 1]
            merged_outputs[output_key] = torch.cat([row for row in row_outputs if row is not None], dim=0)
        return merged_outputs

    def _forward_micro_batch_without_ephemeral_lora(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        sft_category_global_step = int(data.meta_info.get("luffy_global_step", 0))
        sft_category_weights = resolve_sft_category_weights(self.config, sft_category_global_step)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if (
            self.config.get("use_sft_multitask_loss", False)
            or self.config.get("use_off_policy_loss", False)
        ) and "prefix_mask" in data.batch.keys():
            select_keys.append("prefix_mask")
        if self.config.get("use_off_policy_loss", False) and "target_probs" in data.batch.keys():
            select_keys.append("target_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        for key in (
            "ephemeral_lora_path",
            "ephemeral_lora_loss_path",
            self.ephemeral_lora_loss_path_key,
            "ephemeral_lora_variant",
            "lora_variant",
            "luffy_teacher_lora_variant",
            "luffy_teacher_sample_type",
            "update_type",
            "extra_info",
        ):
            if key in data.non_tensor_batch.keys() and key not in non_tensor_select_keys:
                non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    lora_loss_paths = extract_ephemeral_lora_paths_from_micro_batch(
                        model_inputs,
                        batch_size=int(model_inputs["responses"].shape[0]),
                        path_key=self.ephemeral_lora_loss_path_key,
                    )
                    active_lora_loss_paths = [path for path in lora_loss_paths if path is not None]
                    if active_lora_loss_paths:
                        micro_batch_metrics["actor/ephemeral_lora_loss_rows"] = float(len(active_lora_loss_paths))
                        micro_batch_metrics["actor/ephemeral_lora_loss_groups"] = float(len(set(active_lora_loss_paths)))
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    prefix_mask = model_inputs.get("prefix_mask", None)
                    use_prefix_mix = prefix_mask is not None and prefix_mask.numel() > 0
                    use_sft_multitask = self.config.get("use_sft_multitask_loss", False) and use_prefix_mix
                    use_off_policy_loss = self.config.get("use_off_policy_loss", False) and use_prefix_mix
                    if use_sft_multitask and use_off_policy_loss:
                        raise ValueError("use_sft_multitask_loss and use_off_policy_loss cannot both be enabled")
                    if use_sft_multitask:
                        # Keep per-microbatch metric keys rectangular across DP workers.
                        micro_batch_metrics["actor/sft_multitask_loss"] = 0.0
                        micro_batch_metrics["actor/sft_multitask_coef"] = float(self.config.get("sft_loss_coef", 1.0))
                        micro_batch_metrics["actor/off_policy_rows"] = 0
                        for variant_name in SFT_LORA_VARIANTS:
                            micro_batch_metrics[f"actor/sft_multitask_loss/{variant_name}"] = 0.0
                            micro_batch_metrics[f"actor/sft_multitask_rows/{variant_name}"] = 0.0
                        for category in SFT_CATEGORIES:
                            micro_batch_metrics[f"actor/sft_category_loss/{category}"] = 0.0
                            micro_batch_metrics[f"actor/sft_category_rows/{category}"] = 0.0
                            micro_batch_metrics[f"actor/sft_category_nll_sum/{category}"] = 0.0
                            micro_batch_metrics[f"actor/sft_category_token_count/{category}"] = 0.0
                            micro_batch_metrics[f"actor/sft_category_weight/{category}"] = float(
                                sft_category_weights[category]
                            )
                            micro_batch_metrics[f"actor/sft_category_weighted_nll_sum/{category}"] = 0.0
                        micro_batch_metrics["actor/sft_category_weighted_loss"] = 0.0
                        micro_batch_metrics["actor/sft_category_all_teacher_tokens"] = 0.0

                    if use_off_policy_loss:
                        target_probs = model_inputs.get("target_probs", None)
                        if not self.config.get("use_off_policy_probs", False):
                            target_probs = None
                        off_pg_metrics = compute_token_on_off_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=self.config.clip_ratio,
                            clip_upper_bound=float(self.config.get("clip_upper_bound", 1.0)),
                            prefix_mask=prefix_mask,
                            off_cliprange=float(self.config.get("off_policy_cliprange", self.config.clip_ratio)),
                            off_normalize=bool(self.config.get("off_policy_normalize", False)),
                            off_abs_cliprange=(
                                None
                                if self.config.get("use_off_policy_clip", False) is False
                                else float(self.config.get("off_policy_cliprange", self.config.clip_ratio))
                            ),
                            off_max_clip=(
                                None
                                if float(self.config.get("off_policy_max_clip", -1)) < 0
                                else float(self.config.get("off_policy_max_clip"))
                            ),
                            off_min_clip=(
                                None
                                if float(self.config.get("off_policy_min_clip", -1)) < 0
                                else float(self.config.get("off_policy_min_clip"))
                            ),
                            all_max_clip=(
                                None
                                if float(self.config.get("all_max_clip", -1)) < 0
                                else float(self.config.get("all_max_clip"))
                            ),
                            off_policy_reshape=str(self.config.get("off_policy_reshape", "no_reshape")),
                            off_policy_reshape_weight=float(self.config.get("off_policy_reshape_weight", 0.1)),
                            off_policy_reshape_pow_exp=float(self.config.get("off_policy_reshape_pow_exp", 0.5)),
                            off_policy_loss_coef=float(self.config.get("off_policy_loss_coef", 1.0)),
                            on_policy_reshape=str(self.config.get("on_policy_reshape", "no_reshape")),
                            on_policy_reshape_weight=float(self.config.get("on_policy_reshape_weight", 0.1)),
                            on_policy_reshape_pow_exp=float(self.config.get("on_policy_reshape_pow_exp", 0.5)),
                            on_policy_loss_mode=loss_mode,
                            tau_pos=float(self.config.get("tau_pos", 1.0)),
                            tau_neg=float(self.config.get("tau_neg", 1.05)),
                            target_probs=target_probs,
                            loss_remove_token_mean=bool(self.config.get("loss_remove_token_mean", False)),
                            loss_remove_clip=bool(self.config.get("loss_remove_clip", False)),
                        )
                        pg_loss = off_pg_metrics["pg_loss"]
                        policy_loss = pg_loss
                        micro_batch_metrics["actor/on_pg_loss"] = off_pg_metrics["on_pg_loss"].detach().item()
                        micro_batch_metrics["actor/off_pg_loss"] = off_pg_metrics["off_pg_loss"].detach().item()
                        micro_batch_metrics["actor/off_pg_loss_weighted"] = off_pg_metrics["off_pg_loss_weighted"].detach().item()
                        micro_batch_metrics["actor/off_policy_loss_coef"] = off_pg_metrics["off_policy_loss_coef"].detach().item()
                        micro_batch_metrics["actor/pg_clipfrac"] = off_pg_metrics["on_pg_clipfrac"].detach().item()
                        micro_batch_metrics["actor/off_pg_clipfrac"] = off_pg_metrics["off_pg_clipfrac"].detach().item()
                        micro_batch_metrics["actor/ppo_kl"] = off_pg_metrics["ppo_kl"].detach().item()
                        micro_batch_metrics["actor/off_policy_prob"] = off_pg_metrics["off_policy_prob"].detach().item()
                        micro_batch_metrics["actor/on_policy_prob"] = off_pg_metrics["on_policy_prob"].detach().item()
                        micro_batch_metrics["actor/off_ratio_mean"] = off_pg_metrics["off_ratio_mean"].detach().item()
                        micro_batch_metrics["actor/off_ratio_max_clip_frac"] = off_pg_metrics["off_ratio_max_clip_frac"].detach().item()
                        micro_batch_metrics["actor/off_ratio_min_clip_frac"] = off_pg_metrics["off_ratio_min_clip_frac"].detach().item()
                        micro_batch_metrics["actor/off_policy_rows"] = int(prefix_mask.any(dim=-1).sum().item())
                        entropy_mask = response_mask
                        entropy_values = entropy if calculate_entropy else None
                        kl_mask = response_mask
                        kl_log_prob = log_prob
                        kl_ref_log_prob = model_inputs["ref_log_prob"] if self.config.use_kl_loss else None
                    elif use_sft_multitask:
                        off_policy_mask = prefix_mask.any(dim=-1)
                        on_policy_mask = ~off_policy_mask
                        pg_loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
                        sft_loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
                        pg_metrics = {}

                        if on_policy_mask.any():
                            on_rollout_is_weights = (
                                rollout_is_weights[on_policy_mask] if rollout_is_weights is not None else None
                            )
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob[on_policy_mask],
                                log_prob=log_prob[on_policy_mask],
                                advantages=advantages[on_policy_mask],
                                response_mask=response_mask[on_policy_mask],
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=on_rollout_is_weights,
                            )
                            if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                                rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                    log_prob=log_prob[on_policy_mask],
                                    rollout_log_prob=rollout_log_prob[on_policy_mask],
                                    response_mask=response_mask[on_policy_mask],
                                )
                                micro_batch_metrics.update(rollout_corr_metrics)

                        if off_policy_mask.any():
                            category_values = _select_sft_category_values(
                                model_inputs,
                                int(log_prob.shape[0]),
                            )
                            if bool(self.config.get("use_sft_category_loss_weighting", False)):
                                sft_loss, category_metrics = compute_sft_category_weighted_loss(
                                    log_prob=log_prob,
                                    eos_mask=response_mask,
                                    prefix_mask=prefix_mask,
                                    category_values=category_values,
                                    category_weights=sft_category_weights,
                                )
                            else:
                                sft_loss = compute_sft_pure_loss(
                                    log_prob=log_prob[off_policy_mask],
                                    eos_mask=response_mask[off_policy_mask],
                                )
                                _, category_metrics = compute_sft_category_weighted_loss(
                                    log_prob=log_prob,
                                    eos_mask=response_mask,
                                    prefix_mask=prefix_mask,
                                    category_values=category_values,
                                    category_weights={category: 1.0 for category in SFT_CATEGORIES},
                                )
                            micro_batch_metrics["actor/sft_multitask_loss"] = sft_loss.detach().item()
                            micro_batch_metrics["actor/off_policy_rows"] = int(off_policy_mask.sum().item())
                            micro_batch_metrics.update(category_metrics)
                            micro_batch_metrics.update(
                                compute_sft_variant_losses(
                                    log_prob=log_prob,
                                    eos_mask=response_mask,
                                    prefix_mask=prefix_mask,
                                    variant_values=_select_lora_variant_values(model_inputs, int(log_prob.shape[0])),
                                )
                            )

                        micro_batch_metrics.update(pg_metrics)
                        teacher_loss_coef = float(self.config.get("sft_loss_coef", 1.0))
                        policy_loss = pg_loss + teacher_loss_coef * sft_loss
                        micro_batch_metrics["actor/sft_multitask_coef"] = teacher_loss_coef
                        entropy_mask = response_mask
                        entropy_values = entropy if calculate_entropy else None
                        kl_mask = response_mask
                        kl_log_prob = log_prob
                        kl_ref_log_prob = model_inputs["ref_log_prob"] if self.config.use_kl_loss else None
                    else:
                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                        # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                        if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                        policy_loss = pg_loss
                        entropy_mask = response_mask
                        entropy_values = entropy if calculate_entropy else None
                        kl_mask = response_mask
                        kl_log_prob = log_prob
                        kl_ref_log_prob = model_inputs["ref_log_prob"] if self.config.use_kl_loss else None

                    if calculate_entropy and entropy_values is not None and entropy_mask is not None:
                        entropy_agg = agg_loss(loss_mat=entropy_values, loss_mask=entropy_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss and kl_log_prob is not None and kl_mask is not None:
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=kl_log_prob, ref_logprob=kl_ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=kl_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if self.config.get("use_sft_multitask_loss", False):
            # Fallback for unusual empty-worker cases; normal microbatches fill these above.
            metrics.setdefault("actor/sft_multitask_loss", [0.0])
            metrics.setdefault("actor/sft_multitask_coef", [float(self.config.get("sft_loss_coef", 1.0))])
            metrics.setdefault("actor/off_policy_rows", [0.0])
            for variant_name in SFT_LORA_VARIANTS:
                metrics.setdefault(f"actor/sft_multitask_loss/{variant_name}", [0.0])
                metrics.setdefault(f"actor/sft_multitask_rows/{variant_name}", [0.0])
        if self.config.get("use_off_policy_loss", False):
            metrics.setdefault("actor/off_policy_rows", 0.0)
            metrics.setdefault("actor/off_policy_loss_coef", float(self.config.get("off_policy_loss_coef", 1.0)))
            metrics.setdefault("actor/off_pg_loss", 0.0)
            metrics.setdefault("actor/off_pg_loss_weighted", 0.0)
            metrics.setdefault("actor/on_pg_loss", 0.0)
        self.actor_optimizer.zero_grad()
        return metrics
