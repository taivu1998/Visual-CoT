"""
Model Factory.

Supports both Unsloth (optimized, requires Python 3.10+ and CUDA)
and standard transformers (fallback).
"""
import json
import os
from typing import Tuple, Any
import warnings

# Check if Unsloth is available
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Unsloth not available. Using standard transformers. "
        "For optimal performance, use Python 3.10+ with CUDA and install Unsloth."
    )


def _normalize_model_id(model_id: str) -> str:
    if "unsloth/" in model_id:
        return model_id.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "")
    return model_id


def _has_dependency(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _build_quantization_config(enable_4bit: bool = True):
    if not enable_4bit:
        return None

    if not _has_dependency("bitsandbytes"):
        return None

    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _load_processor(model_path: str, fallback_model_id: str):
    from transformers import AutoProcessor

    processor_source = model_path if os.path.exists(model_path) else fallback_model_id
    try:
        return AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    except Exception:
        return AutoProcessor.from_pretrained(fallback_model_id, trust_remote_code=True)


def _read_adapter_base_model(adapter_dir: str) -> str:
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"adapter_config.json in '{adapter_dir}' does not include base_model_name_or_path")
    return _normalize_model_id(base_model_name)


def _load_peft_inference_model(model_path: str):
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq

    base_model_id = _read_adapter_base_model(model_path)
    quantization_config = _build_quantization_config(enable_4bit=True)

    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    processor = _load_processor(model_path, base_model_id)
    model.eval()
    return model, processor, "transformers-peft"


def load_model_unsloth(config: dict) -> Tuple[Any, Any]:
    """Load model using Unsloth (optimized)."""
    from unsloth import FastVisionModel

    model_cfg = config["model"]
    lora_cfg = config["lora"]

    model, tokenizer = FastVisionModel.from_pretrained(
        model_cfg["base_model_id"],
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
    )

    return model, tokenizer


def load_model_transformers(config: dict) -> Tuple[Any, Any]:
    """Load model using standard transformers + PEFT (fallback)."""
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    model_cfg = config["model"]
    lora_cfg = config["lora"]

    # Use a compatible model ID for transformers
    model_id = _normalize_model_id(model_cfg["base_model_id"])

    # Quantization config
    bnb_config = _build_quantization_config(enable_4bit=model_cfg.get("load_in_4bit", True))

    # Load model and processor
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Prepare for training
    if model_cfg.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model)

    # Add LoRA
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def load_model(config: dict, force_transformers: bool = False) -> Tuple[Any, Any]:
    """
    Loads the model with LoRA adapters.

    Uses Unsloth if available (faster), otherwise falls back to transformers.

    Args:
        config: Configuration dictionary
        force_transformers: If True, use transformers even if Unsloth is available

    Returns:
        Tuple of (model, tokenizer/processor)
    """
    if UNSLOTH_AVAILABLE and not force_transformers:
        print("Loading model with Unsloth (optimized)...")
        return load_model_unsloth(config)
    else:
        print("Loading model with transformers + PEFT (fallback)...")
        return load_model_transformers(config)


def load_model_for_inference(model_path: str, prefer_unsloth: bool = True) -> Tuple[Any, Any, str]:
    """
    Load a trained checkpoint or Hugging Face model for inference.
    """
    from transformers import AutoModelForVision2Seq

    if prefer_unsloth and UNSLOTH_AVAILABLE:
        from unsloth import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
        FastVisionModel.for_inference(model)
        return model, processor, "unsloth"

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        if not _has_dependency("peft"):
            raise RuntimeError(
                f"Inference checkpoint '{model_path}' is a PEFT adapter checkpoint, but 'peft' is not installed."
            )
        return _load_peft_inference_model(model_path)

    model_id = _normalize_model_id(model_path)
    quantization_config = _build_quantization_config(enable_4bit=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = _load_processor(model_path, model_id)
    model.eval()
    backend_name = "transformers-4bit" if quantization_config is not None else "transformers"
    return model, processor, backend_name
