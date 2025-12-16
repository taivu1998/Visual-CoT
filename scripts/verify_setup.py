#!/usr/bin/env python
"""
V-CoT Setup Verification Script

Checks that all dependencies and configurations are properly set up
before running experiments.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARN = f"{YELLOW}!{RESET}"


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 8):
        print(f"  {CHECK} Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  {CROSS} Python {version.major}.{version.minor} (requires >= 3.8)")
        return False


def check_package(name, import_name=None):
    """Check if a package is installed."""
    import_name = import_name or name
    try:
        __import__(import_name)
        print(f"  {CHECK} {name}")
        return True
    except ImportError:
        print(f"  {CROSS} {name} (not installed)")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  {CHECK} CUDA available: {device_name}")
            return True
        else:
            print(f"  {WARN} CUDA not available (will use CPU - very slow)")
            return False
    except Exception as e:
        print(f"  {CROSS} Could not check CUDA: {e}")
        return False


def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"  {CHECK} {description}: {path}")
        return True
    else:
        print(f"  {CROSS} {description}: {path} (not found)")
        return False


def check_directory(path, description):
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"  {CHECK} {description}: {path}")
        return True
    else:
        print(f"  {CROSS} {description}: {path} (not found)")
        return False


def check_env_var(name, required=False):
    """Check if an environment variable is set."""
    value = os.environ.get(name)
    if value:
        # Mask the value for security
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
        print(f"  {CHECK} {name}: {masked}")
        return True
    else:
        if required:
            print(f"  {CROSS} {name} (not set - required)")
        else:
            print(f"  {WARN} {name} (not set - optional)")
        return not required


def check_data_files():
    """Check for training data files."""
    train_path = Path("data/processed/train.jsonl")
    val_path = Path("data/processed/val.jsonl")
    sample_train = Path("data/processed/sample_train.jsonl")
    sample_val = Path("data/processed/sample_val.jsonl")

    has_full_data = train_path.exists() and val_path.exists()
    has_sample_data = sample_train.exists() and sample_val.exists()

    if has_full_data:
        print(f"  {CHECK} Full training data available")
        return True
    elif has_sample_data:
        print(f"  {WARN} Only sample data available (run 'make generate' for full data)")
        return True
    else:
        print(f"  {CROSS} No training data found")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("V-CoT Setup Verification")
    print("=" * 60)

    all_passed = True

    # Python version
    print("\n[1/6] Python Environment:")
    all_passed &= check_python_version()

    # Core packages
    print("\n[2/6] Core Dependencies:")
    core_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("accelerate", "accelerate"),
        ("pyyaml", "yaml"),
    ]
    for name, import_name in core_packages:
        all_passed &= check_package(name, import_name)

    # Optional packages
    print("\n[3/6] Optional Dependencies:")
    optional_packages = [
        ("unsloth", "unsloth"),
        ("bitsandbytes", "bitsandbytes"),
        ("gradio", "gradio"),
        ("openai", "openai"),
        ("opencv-python", "cv2"),
    ]
    for name, import_name in optional_packages:
        check_package(name, import_name)  # Don't fail on optional

    # CUDA
    print("\n[4/6] GPU Support:")
    check_cuda()  # Warning only, don't fail

    # Configuration files
    print("\n[5/6] Configuration Files:")
    all_passed &= check_file("configs/default.yaml", "Default config")
    check_file("configs/test.yaml", "Test config")  # Optional

    # Data
    print("\n[6/6] Training Data:")
    check_data_files()

    # Environment variables
    print("\n[Optional] Environment Variables:")
    check_env_var("OPENAI_API_KEY")
    check_env_var("WANDB_API_KEY")
    check_env_var("HF_TOKEN")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN}Setup verification PASSED{RESET}")
        print("\nNext steps:")
        print("  1. Install remaining dependencies: make install")
        print("  2. Generate training data: make generate")
        print("  3. Train the model: make train")
        print("  4. Launch demo: make demo")
    else:
        print(f"{RED}Setup verification FAILED{RESET}")
        print("\nPlease fix the issues above before proceeding.")
        print("Run 'make install' to install missing dependencies.")

    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
