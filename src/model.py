"""
Model Factory.

Supports both Unsloth (optimized, requires Python 3.10+ and CUDA)
and standard transformers (fallback).
"""
from typing import Tuple, Any, Optional
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
    model_id = model_cfg["base_model_id"]
    if "unsloth" in model_id:
        # Convert unsloth model ID to standard HF model ID
        model_id = model_id.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "")

    # Quantization config
    bnb_config = None
    if model_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

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