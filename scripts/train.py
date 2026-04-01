"""
Main entry point for training.

Training modes:
- multimodal: default and recommended
- text_only_debug: lightweight smoke test path
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.model import load_model
from src.trainer import VCoTTrainer
from src.utils import seed_everything, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="V-CoT Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--mode",
        choices=["multimodal", "text_only_debug"],
        default="multimodal",
        help="Training mode. 'multimodal' is the real VLM path. 'text_only_debug' is a smoke test path.",
    )
    parser.add_argument(
        "--allow-new-config-keys",
        action="store_true",
        help="Allow dot-notation overrides to create new keys instead of failing on unknown paths.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, override_tokens = parse_args()
    config = load_config(
        config_path=args.config,
        override_tokens=override_tokens,
        allow_new_keys=args.allow_new_config_keys,
    )

    logger = setup_logger("VCoT-Train")
    seed_everything(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("V-CoT TRAINING")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Model: %s", config["model"]["base_model_id"])
    logger.info("Mode: %s", args.mode)
    logger.info("Resume: %s", args.resume)
    logger.info("=" * 60)

    model, processor = load_model(config)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    trainer = VCoTTrainer.from_files(
        model=model,
        processor=processor,
        train_file=config["data"]["train_path"],
        config=config,
        val_file=config["data"].get("val_path"),
        mode=args.mode,
        repo_root=repo_root,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
