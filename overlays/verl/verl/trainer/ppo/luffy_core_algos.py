from __future__ import annotations

import torch

import verl.utils.torch_functional as verl_F


def compute_sft_pure_loss(log_prob: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
    sft_losses = -log_prob
    return verl_F.masked_mean(sft_losses, eos_mask)


def compute_token_on_off_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    clip_upper_bound: float,
    prefix_mask: torch.Tensor,
    off_cliprange: float,
    off_normalize: bool = False,
    off_abs_cliprange=None,
    off_max_clip=None,
    off_min_clip=None,
    all_max_clip=None,
    off_policy_reshape: str = "no_reshape",
    off_policy_reshape_weight: float = 1.0,
    off_policy_reshape_pow_exp: float = 0.5,
    off_policy_loss_coef: float = 1.0,
    on_policy_reshape: str = "no_reshape",
    on_policy_reshape_weight: float = 1.0,
    on_policy_reshape_pow_exp: float = 0.5,
    on_policy_loss_mode: str = "vanilla",
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
    target_probs: torch.Tensor | None = None,
    loss_remove_token_mean: bool = False,
    loss_remove_clip: bool = False,
) -> dict[str, torch.Tensor]:
    negative_approx_kl = log_prob - old_log_prob
    on_policy_loss_mode = str(on_policy_loss_mode).strip().lower()
    sapo_negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl_source = sapo_negative_approx_kl if on_policy_loss_mode == "sapo" else negative_approx_kl
    ppo_kl = verl_F.masked_mean(-ppo_kl_source, eos_mask)

    prefix_mask = prefix_mask.to(dtype=torch.bool)
    eos_mask_bool = eos_mask.to(dtype=torch.bool)
    on_mask = (~prefix_mask) & eos_mask_bool
    off_mask = prefix_mask & eos_mask_bool

    if on_policy_loss_mode == "sapo":
        ratio = torch.exp(sapo_negative_approx_kl)
        tau_pos_tensor = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
        tau_neg_tensor = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)
        taus = torch.where(advantages > 0, tau_pos_tensor, tau_neg_tensor)
        gates = torch.sigmoid(taus * (ratio - 1.0)) * (4.0 / taus)
        on_pg_losses = -gates * advantages
        on_pg_clipfrac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
    else:
        if on_policy_reshape == "no_reshape":
            ratio = torch.exp(negative_approx_kl)
        elif on_policy_reshape == "logp":
            ratio = log_prob - old_log_prob
        elif on_policy_reshape == "p_logp":
            ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
        elif on_policy_reshape == "square_root":
            ratio = torch.sqrt(torch.exp(negative_approx_kl))
        elif on_policy_reshape == "pow":
            ratio = torch.pow(torch.exp(negative_approx_kl), on_policy_reshape_pow_exp)
        elif on_policy_reshape == "p_div_p_0.1":
            prob = torch.exp(log_prob)
            old_prob = torch.exp(old_log_prob)
            ratio = (prob / (prob + 0.1)) / (old_prob / (old_prob + 0.1))
        elif on_policy_reshape == "p_div_p_0.5":
            prob = torch.exp(log_prob)
            old_prob = torch.exp(old_log_prob)
            ratio = (prob / (prob + 0.5)) / (old_prob / (old_prob + 0.5))
        else:
            raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

        on_pg_losses = -advantages * ratio
        upper_bound = max(float(clip_upper_bound), 1.0 + float(cliprange))
        if loss_remove_clip:
            on_pg_clipfrac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        else:
            on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
            on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
            on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, on_mask)
    if torch.isnan(on_pg_loss).item():
        on_pg_loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)

    if on_policy_loss_mode == "sapo":
        if target_probs is None:
            off_log_ratio = torch.clamp(log_prob, min=-20.0, max=20.0)
        else:
            if target_probs.shape != log_prob.shape:
                raise ValueError(f"target_probs shape mismatch: expected {log_prob.shape}, got {target_probs.shape}")
            off_log_ratio = torch.clamp(log_prob - torch.log(target_probs + 1e-6), min=-20.0, max=20.0)
        off_ratio = torch.exp(off_log_ratio) * prefix_mask
        off_ratio_mean = verl_F.masked_mean(off_ratio, off_mask)
        if torch.isnan(off_ratio_mean).item():
            off_ratio_mean = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        off_taus = torch.where(advantages > 0, tau_pos_tensor, tau_neg_tensor)
        off_gates = torch.sigmoid(off_taus * (off_ratio - 1.0)) * (4.0 / off_taus)
        off_pg_losses = -off_gates * advantages
        off_pg_loss = verl_F.masked_mean(off_pg_losses, off_mask)
        if torch.isnan(off_pg_loss).item():
            off_pg_loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        off_pg_clipfrac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        off_ratio_max_clip_frac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        off_ratio_min_clip_frac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
    else:
        if target_probs is None:
            off_ratio = torch.exp(log_prob)
            if off_policy_reshape == "no_reshape":
                pass
            elif off_policy_reshape == "logp":
                off_ratio = log_prob * off_policy_reshape_weight
            elif off_policy_reshape == "p_logp":
                off_ratio = log_prob * off_policy_reshape_weight + off_ratio
            elif off_policy_reshape == "square_root":
                off_ratio = torch.sqrt(off_ratio)
            elif off_policy_reshape == "p_div_p_0.1":
                off_ratio = off_ratio / (off_ratio + 0.1)
            elif off_policy_reshape == "p_div_p_0.5":
                off_ratio = off_ratio / (off_ratio + 0.5)
            elif off_policy_reshape == "p_div_p_0.3":
                off_ratio = off_ratio / (off_ratio + 0.3)
            elif off_policy_reshape == "pow":
                off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
            else:
                raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
        else:
            if target_probs.shape != log_prob.shape:
                raise ValueError(f"target_probs shape mismatch: expected {log_prob.shape}, got {target_probs.shape}")
            off_ratio = torch.exp(log_prob) / (target_probs + 1e-6)
            off_ratio = off_ratio * prefix_mask

        if off_normalize:
            off_mean = verl_F.masked_mean(off_ratio, off_mask)
            if not torch.isnan(off_mean).item() and off_mean.abs().item() > 0:
                off_ratio = off_ratio / off_mean

        if off_abs_cliprange is not None and off_abs_cliprange >= 0:
            off_ratio = torch.clamp(off_ratio, max=off_abs_cliprange)

        if off_max_clip is not None:
            off_ratio = torch.clamp(off_ratio, max=off_max_clip)
            off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), off_mask)
        else:
            off_ratio_max_clip_frac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)

        if off_min_clip is not None:
            off_ratio = torch.clamp(off_ratio, min=off_min_clip)
            off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), off_mask)
        else:
            off_ratio_min_clip_frac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)

        if off_cliprange is not None and off_cliprange >= 0:
            off_ratio = torch.clamp(off_ratio, max=1.0 + off_cliprange)

        off_ratio_mean = verl_F.masked_mean(off_ratio, off_mask)
        if torch.isnan(off_ratio_mean).item():
            off_ratio_mean = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)

        off_pg_losses = -advantages * off_ratio
        off_pg_loss = verl_F.masked_mean(off_pg_losses, off_mask)
        if torch.isnan(off_pg_loss).item():
            off_pg_loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        off_pg_clipfrac = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)

    off_policy_loss_coef_tensor = torch.as_tensor(
        max(float(off_policy_loss_coef), 0.0),
        dtype=log_prob.dtype,
        device=log_prob.device,
    )
    prefix_mask_f = prefix_mask.float()
    weighted_off_pg_losses = off_pg_losses * off_policy_loss_coef_tensor
    pg_losses = weighted_off_pg_losses * prefix_mask_f + on_pg_losses * (1 - prefix_mask_f)

    off_policy_prob = verl_F.masked_mean(torch.exp(log_prob), off_mask)
    if torch.isnan(off_policy_prob).item():
        off_policy_prob = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
    on_policy_prob = verl_F.masked_mean(torch.exp(old_log_prob), on_mask)
    if torch.isnan(on_policy_prob).item():
        on_policy_prob = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)

    if all_max_clip is not None:
        on_prob = torch.exp(log_prob)
        keep_mask = (on_prob <= all_max_clip).to(dtype=eos_mask.dtype)
        eos_mask = eos_mask * keep_mask
        pg_losses = pg_losses * keep_mask

    if loss_remove_token_mean:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "off_pg_loss_weighted": off_pg_loss * off_policy_loss_coef_tensor,
        "on_pg_loss": on_pg_loss,
        "off_policy_loss_coef": off_policy_loss_coef_tensor,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }
