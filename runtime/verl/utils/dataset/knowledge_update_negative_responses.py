from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _build_templates(starts: Iterable[str], endings: Iterable[str], *, limit: int = 128) -> list[str]:
    templates: list[str] = []
    seen: set[str] = set()
    for start in starts:
        for ending in endings:
            text = f"{start} {ending}".strip()
            if text in seen:
                continue
            seen.add(text)
            templates.append(text)
            if len(templates) >= limit:
                return templates
    return templates


_NO_CHANGE_STARTS = [
    "No update is needed here;",
    "There is no revised fact to apply;",
    "I do not see a changed fact in this case;",
    "The original answer remains applicable;",
    "Nothing in the update signal changes the answer;",
    "There is no new information to incorporate;",
    "The fact appears unchanged;",
    "No factual correction is provided;",
    "The available signal does not alter the answer;",
    "This case does not contain an update;",
    "I should keep the answer unchanged;",
    "The evidence does not introduce a replacement fact;",
    "No update has been supplied for this query;",
    "The answer should stay as it was;",
    "There is no actionable factual change;",
    "The provided information leaves the answer unchanged;",
]

_NO_CHANGE_ENDINGS = [
    "the prior answer should be retained.",
    "the existing answer is still the appropriate response.",
    "I would answer with the unchanged fact.",
    "there is no updated fact to report.",
    "the original fact should continue to be used.",
    "I should not invent a different answer.",
    "the unchanged answer is the reliable choice.",
    "the response should reflect that no change occurred.",
]

_RANDOM_CHANGE_STARTS = [
    "I cannot reliably determine the updated fact because",
    "The update signal is not trustworthy because",
    "I should not provide an updated fact because",
    "The proposed update is unusable because",
    "A reliable answer cannot be derived because",
    "The updated fact cannot be recovered because",
    "I cannot resolve the intended correction because",
    "The evidence is insufficient for an update because",
    "The signal should be treated as inconsistent because",
    "I should reject the update attempt because",
    "The update path is ambiguous because",
    "No dependable revised answer is available because",
    "I cannot select a corrected fact because",
    "The provided update should not be applied because",
    "The answer cannot be updated safely because",
    "The correction signal is unreliable because",
]

_RANDOM_CHANGE_ENDINGS = [
    "the update signal is inconsistent.",
    "the supplied evidence conflicts with itself.",
    "the adapter signal does not match a coherent fact.",
    "the retrieved update appears randomly mismatched.",
    "the available signals point to incompatible answers.",
    "the modification cue is noisy and contradictory.",
    "the source of the change is not internally consistent.",
    "the requested update is confused with an unrelated signal.",
]

DEFAULT_NO_CHANGE_RESPONSES = _build_templates(_NO_CHANGE_STARTS, _NO_CHANGE_ENDINGS)
DEFAULT_RANDOM_CHANGE_RESPONSES = _build_templates(_RANDOM_CHANGE_STARTS, _RANDOM_CHANGE_ENDINGS)


def _contains_cjk(text: str) -> bool:
    return any(
        ("\u4e00" <= char <= "\u9fff")
        or ("\u3400" <= char <= "\u4dbf")
        or ("\uf900" <= char <= "\ufaff")
        for char in text
    )


def validate_response_templates(values: Iterable[Any], *, name: str) -> list[str]:
    templates: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        text = str(raw_value).strip()
        if not text:
            continue
        if _contains_cjk(text):
            raise ValueError(f"{name} response template contains CJK text: {text!r}")
        if text in seen:
            raise ValueError(f"{name} response template is duplicated: {text!r}")
        seen.add(text)
        templates.append(text)
    if len(templates) < 128:
        raise ValueError(f"{name} requires at least 128 non-empty unique English response templates.")
    return templates


def load_response_templates(path: Any, *, defaults: list[str], name: str) -> list[str]:
    if path in (None, "", "None"):
        return validate_response_templates(defaults, name=name)
    template_path = Path(str(path)).expanduser()
    with template_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in (name, "responses", "templates"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError(f"{name} response template file must contain a JSON list or an object with a list value.")
    return validate_response_templates(payload, name=name)
