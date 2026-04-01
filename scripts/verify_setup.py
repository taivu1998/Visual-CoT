#!/usr/bin/env python
"""
V-CoT setup verification.

Validates the actual repo workflows instead of checking mismatched paths.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runtime import detect_runtime_availability

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARN = f"{YELLOW}!{RESET}"


def print_status(ok: bool, label: str, details: str = ""):
    prefix = CHECK if ok else CROSS
    suffix = f": {details}" if details else ""
    print(f"  {prefix} {label}{suffix}")


def print_warn(label: str, details: str = ""):
    suffix = f": {details}" if details else ""
    print(f"  {WARN} {label}{suffix}")


def check_python_matrix():
    version = sys.version_info
    if version >= (3, 10):
        print_status(True, "Python", f"{version.major}.{version.minor}.{version.micro} (recommended for Unsloth)")
        return True
    if version >= (3, 8):
        print_warn("Python", f"{version.major}.{version.minor}.{version.micro} (core repo may work, Unsloth likely will not)")
        return True

    print_status(False, "Python", f"{version.major}.{version.minor} (requires >= 3.8)")
    return False


def check_path(path: str, description: str) -> bool:
    exists = Path(path).exists()
    print_status(exists, description, path)
    return exists


def detect_dataset_status() -> bool:
    canonical_paths = [
        Path("data/processed/train.jsonl"),
        Path("data/processed/val.jsonl"),
    ]
    sample_paths = [
        Path("data/processed/sample_train.jsonl"),
        Path("data/processed/sample_val.jsonl"),
    ]

    if all(path.exists() for path in canonical_paths):
        print_status(True, "Canonical processed data", "data/processed/train.jsonl + data/processed/val.jsonl")
        return True
    if all(path.exists() for path in sample_paths):
        print_warn("Only sample data found", "data/processed/sample_train.jsonl + data/processed/sample_val.jsonl")
        return True

    print_status(False, "Processed data", "missing canonical or sample processed datasets")
    return False


def main():
    availability = detect_runtime_availability()

    print("\n" + "=" * 60)
    print("V-CoT Setup Verification")
    print("=" * 60)

    all_passed = True

    print("\n[1/6] Python:")
    all_passed &= check_python_matrix()

    print("\n[2/6] Canonical Paths:")
    all_passed &= check_path("configs/default.yaml", "Default config")
    check_path("configs/test.yaml", "Test config")
    detect_dataset_status()

    print("\n[3/6] Core Dependencies:")
    core_checks = {
        "torch": availability.torch,
        "transformers": availability.transformers,
        "datasets": availability.datasets,
        "peft": availability.peft,
        "accelerate": availability.accelerate,
    }
    for name, ok in core_checks.items():
        all_passed &= ok
        print_status(ok, name)

    print("\n[4/6] Optional Workflow Dependencies:")
    print_status(availability.bitsandbytes, "bitsandbytes 4-bit loading")
    print_status(availability.unsloth, "Unsloth backend")
    print_status(availability.gradio, "Gradio demo")
    print_status(availability.qwen_vl_utils, "qwen_vl_utils preprocessing")
    print_status(availability.openai, "OpenAI ScienceQA generation")

    print("\n[5/6] Workflow Readiness:")
    training_ready = all([availability.torch, availability.transformers, availability.peft, availability.accelerate])
    inference_ready = all([availability.torch, availability.transformers])
    demo_ready = inference_ready and availability.gradio
    generation_ready = availability.datasets
    scienceqa_ready = generation_ready and availability.openai and bool(os.environ.get("OPENAI_API_KEY"))

    print_status(training_ready, "Training", "core training stack")
    print_status(inference_ready, "Inference", "single-image and evaluation stack")
    if inference_ready and not availability.bitsandbytes:
        print_warn("Inference quantization", "bitsandbytes is not installed, so transformer inference will use non-4-bit loading")
    print_status(demo_ready, "Demo", "Gradio + inference stack")
    print_status(generation_ready, "VisCOT generation", "datasets package available")
    print_status(scienceqa_ready, "ScienceQA generation", "OpenAI package + OPENAI_API_KEY")

    print("\n[6/6] Environment Variables:")
    if os.environ.get("OPENAI_API_KEY"):
        print_status(True, "OPENAI_API_KEY", "set")
    else:
        print_warn("OPENAI_API_KEY", "not set")
    if os.environ.get("WANDB_API_KEY"):
        print_status(True, "WANDB_API_KEY", "set")
    else:
        print_warn("WANDB_API_KEY", "not set")
    if os.environ.get("HF_TOKEN"):
        print_status(True, "HF_TOKEN", "set")
    else:
        print_warn("HF_TOKEN", "not set")

    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN}Core setup checks passed{RESET}")
        print("Recommended next steps:")
        print("  1. python scripts/validate_data.py --input data/processed/sample_train.jsonl --allow-missing-images")
        print("  2. python scripts/train.py --config configs/test.yaml --mode text_only_debug")
        print("  3. python scripts/train.py --config configs/default.yaml --mode multimodal")
    else:
        print(f"{RED}Core setup checks failed{RESET}")
        print("Install the missing dependencies and re-run this script.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
