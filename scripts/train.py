"""
Main entry point for training.

Supports both traditional and pre-tokenized training approaches.
The pre-tokenized approach (default) is recommended for VLM training
as it avoids issues with multimodal tokenizer collation.
"""
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_parser import ConfigParser
from src.utils import setup_logger, seed_everything
from src.dataset import load_dataset_from_jsonl
from src.model import load_model
from src.trainer import VCoTTrainer


def main():
    arg_parser = argparse.ArgumentParser(description="V-CoT Training")
    arg_parser.add_argument("--config", type=str, default="configs/default.yaml",
                           help="Path to config file")
    arg_parser.add_argument("--resume", action="store_true",
                           help="Resume from latest checkpoint")
    arg_parser.add_argument("--pretokenized", action="store_true", default=True,
                           help="Use pre-tokenized approach (recommended)")
    arg_parser.add_argument("--no_pretokenized", action="store_false", dest="pretokenized",
                           help="Use traditional approach (may have issues with VLM)")
    args = arg_parser.parse_args()

    # Load config
    parser = ConfigParser()
    config = parser.load(args.config if hasattr(args, 'config') else None)

    # Setup
    logger = setup_logger("VCoT-Train")
    seed_everything(config["project"]["seed"])

    logger.info(f"=" * 60)
    logger.info("V-CoT TRAINING")
    logger.info(f"=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {config['model']['base_model_id']}")
    logger.info(f"LoRA rank: {config['lora']['rank']}, alpha: {config['lora']['alpha']}")
    logger.info(f"Pre-tokenized: {args.pretokenized}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"=" * 60)

    # Load model
    model, tokenizer = load_model(config)

    # Create trainer
    if args.pretokenized:
        # Recommended approach: pre-tokenize the data
        logger.info("Using pre-tokenized training approach (recommended)")
        trainer = VCoTTrainer.from_pretokenized_files(
            model=model,
            tokenizer=tokenizer,
            train_file=config["data"]["train_path"],
            config=config,
            val_file=config["data"].get("val_path"),
            max_seq_length=config["model"]["max_seq_length"]
        )
    else:
        # Traditional approach
        logger.info("Using traditional training approach")
        train_dataset = load_dataset_from_jsonl(config["data"]["train_path"])
        val_dataset = None
        if config["data"].get("val_path") and os.path.exists(config["data"]["val_path"]):
            val_dataset = load_dataset_from_jsonl(config["data"]["val_path"])

        trainer = VCoTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset
        )

    # Train
    trainer.train(resume_from_checkpoint=args.resume)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()