"""
Runtime capability detection and optional dependency helpers.
"""
from dataclasses import dataclass
import importlib
from typing import Optional


DEPENDENCY_HINTS = {
    "unsloth": "Install Unsloth for optimized VLM training and inference.",
    "gradio": "Install gradio to launch the demo UI.",
    "qwen_vl_utils": "Install qwen-vl-utils for Qwen image preprocessing.",
    "openai": "Install openai and set OPENAI_API_KEY for ScienceQA generation.",
    "trl": "Install trl if you want TRL-specific training flows.",
}


@dataclass
class RuntimeAvailability:
    torch: bool
    transformers: bool
    datasets: bool
    peft: bool
    accelerate: bool
    bitsandbytes: bool
    unsloth: bool
    gradio: bool
    qwen_vl_utils: bool
    openai: bool
    trl: bool


def _has_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def detect_runtime_availability() -> RuntimeAvailability:
    return RuntimeAvailability(
        torch=_has_module("torch"),
        transformers=_has_module("transformers"),
        datasets=_has_module("datasets"),
        peft=_has_module("peft"),
        accelerate=_has_module("accelerate"),
        bitsandbytes=_has_module("bitsandbytes"),
        unsloth=_has_module("unsloth"),
        gradio=_has_module("gradio"),
        qwen_vl_utils=_has_module("qwen_vl_utils"),
        openai=_has_module("openai"),
        trl=_has_module("trl"),
    )


def require_dependency(module_name: str, feature_name: str, extra_hint: Optional[str] = None) -> None:
    if _has_module(module_name):
        return

    hint = extra_hint or DEPENDENCY_HINTS.get(module_name, "")
    message = f"'{feature_name}' requires the optional dependency '{module_name}'."
    if hint:
        message = f"{message} {hint}"
    raise RuntimeError(message)


def get_process_vision_info():
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info
    except Exception:
        return None
