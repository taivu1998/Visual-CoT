"""
Training Loop Encapsulation.

Wraps the SFTTrainer from TRL with V-CoT specific configuration.
Handles both vision-language (multimodal) and text-only training.
Includes NEFTune, early stopping, and cosine scheduler for optimal training.
"""
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq
import os
import sys
import yaml
from typing import Optional, Tuple
from datasets import Dataset

# Handle imports for both package and script usage
try:
    from src.dataset import (
        prepare_dataset,
        prepare_pretokenized_dataset,
        extract_text_tokenizer
    )
except ImportError:
    from dataset import (
        prepare_dataset,
        prepare_pretokenized_dataset,
        extract_text_tokenizer
    )


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(checkpoint_dir, latest)


class VCoTTrainer:
    """
    Visual Chain-of-Thought Trainer.

    Wraps SFTTrainer with configuration optimized for training
    vision-language models to produce grounded reasoning traces.

    Features:
    - NEFTune: Adds noise to embeddings for better generalization
    - Early Stopping: Stops training when validation loss plateaus
    - Cosine LR Schedule: Smooth learning rate decay
    - Pre-tokenization: Avoids VLM collator issues
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        config: dict,
        val_dataset: Optional[Dataset] = None,
        use_pretokenized: bool = False,
        text_tokenizer=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_args = config["training"]
        self.use_pretokenized = use_pretokenized

        # Extract or use provided text tokenizer for pre-tokenized datasets
        if use_pretokenized:
            self.text_tokenizer = text_tokenizer or extract_text_tokenizer(tokenizer)
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        else:
            # Prepare datasets - convert messages format to text format
            self.text_tokenizer = tokenizer
            self.train_dataset = prepare_dataset(train_dataset, tokenizer)
            self.val_dataset = prepare_dataset(val_dataset, tokenizer) if val_dataset else None

    @classmethod
    def from_pretokenized_files(
        cls,
        model,
        tokenizer,
        train_file: str,
        config: dict,
        val_file: Optional[str] = None,
        max_seq_length: int = 2048
    ) -> "VCoTTrainer":
        """
        Create trainer from pre-tokenized JSONL files.

        This is the recommended approach for VLM training.
        """
        train_dataset, text_tokenizer = prepare_pretokenized_dataset(
            train_file, tokenizer, max_seq_length
        )

        val_dataset = None
        if val_file and os.path.exists(val_file):
            val_dataset, _ = prepare_pretokenized_dataset(
                val_file, tokenizer, max_seq_length
            )

        return cls(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            use_pretokenized=True,
            text_tokenizer=text_tokenizer
        )

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Run the training loop."""
        # Ensure output directory exists
        output_dir = self.train_args["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Check for checkpoint resume
        if resume_from_checkpoint is True:
            resume_from_checkpoint = find_latest_checkpoint(output_dir)
            if resume_from_checkpoint:
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            else:
                print("No checkpoint found, starting fresh training")
                resume_from_checkpoint = None

        # Calculate training steps for scheduler
        total_samples = len(self.train_dataset)
        batch_size = self.train_args["per_device_train_batch_size"]
        grad_accum = self.train_args["gradient_accumulation_steps"]
        effective_batch = batch_size * grad_accum
        steps_per_epoch = total_samples // effective_batch

        # Get epochs or max_steps
        num_epochs = self.train_args.get("num_epochs", None)
        max_steps = self.train_args.get("max_steps", -1)

        if max_steps == -1 and num_epochs:
            total_steps = steps_per_epoch * num_epochs
            print(f"Training for {num_epochs} epochs = {total_steps} steps")
        else:
            total_steps = max_steps
            print(f"Training for {max_steps} steps")

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Effective batch size: {effective_batch}")

        # Use DataCollatorForSeq2Seq for pre-tokenized data
        data_collator = None
        if self.use_pretokenized:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.text_tokenizer,
                padding=True,
                return_tensors="pt",
            )

        # Configure SFT training with improvements
        sft_conf = SFTConfig(
            output_dir=output_dir,

            # Duration
            num_train_epochs=num_epochs if max_steps == -1 else None,
            max_steps=max_steps if max_steps != -1 else -1,

            # Batch size
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,

            # Learning rate with cosine schedule
            learning_rate=float(self.train_args["learning_rate"]),
            lr_scheduler_type=self.train_args.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.train_args.get("warmup_ratio", 0.03),
            warmup_steps=self.train_args.get("warmup_steps", 0),

            # Regularization
            weight_decay=self.train_args.get("weight_decay", 0.01),
            max_grad_norm=self.train_args.get("max_grad_norm", 0.3),

            # NEFTune - adds noise to embeddings for better generalization
            neftune_noise_alpha=self.train_args.get("neftune_noise_alpha", 5),

            # Efficiency
            max_seq_length=self.config["model"]["max_seq_length"],
            packing=False,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            gradient_checkpointing=self.train_args.get("gradient_checkpointing", True),
            optim=self.train_args.get("optimizer", "adamw_8bit"),

            # Evaluation & Early Stopping
            eval_strategy="steps" if self.val_dataset else "no",
            eval_steps=self.train_args.get("eval_steps", max(500, steps_per_epoch // 4)),
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Saving
            save_strategy="steps",
            save_steps=self.train_args.get("save_steps", max(500, steps_per_epoch // 2)),
            save_total_limit=self.train_args.get("save_total_limit", 3),

            # Logging
            logging_steps=self.train_args.get("logging_steps", 50),
            logging_first_step=True,
            report_to=self.config["project"].get("report_to", "tensorboard"),

            # Resume
            resume_from_checkpoint=resume_from_checkpoint,

            # Other
            seed=self.config["project"].get("seed", 42),
            dataloader_num_workers=0,
            remove_unused_columns=True if self.use_pretokenized else False,
        )

        # Setup callbacks
        callbacks = []
        if self.val_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_args.get("early_stopping_patience", 3),
                    early_stopping_threshold=self.train_args.get("early_stopping_threshold", 0.001)
                )
            )

        # Initialize trainer
        trainer_kwargs = {
            "model": self.model,
            "args": sft_conf,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.val_dataset,
            "callbacks": callbacks if callbacks else None,
        }

        if self.use_pretokenized:
            trainer_kwargs["data_collator"] = data_collator
            trainer_kwargs["tokenizer"] = self.text_tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer

        trainer = SFTTrainer(**trainer_kwargs)

        # Run training
        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model and tokenizer
        self._save_checkpoint()

        print(f"\nTraining complete!")
        print(f"Total steps: {trainer_stats.global_step}")
        print(f"Training loss: {trainer_stats.training_loss:.4f}")

        return trainer

    def _save_checkpoint(self):
        """Save model, tokenizer, and training config."""
        output_dir = self.train_args["output_dir"]

        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training configuration for reproducibility
        config_path = os.path.join(output_dir, "training_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"Checkpoint saved to {output_dir}")

    def save_lora_only(self, output_dir: Optional[str] = None):
        """
        Save only the LoRA adapter weights (smaller file size).
        Useful for sharing fine-tuned models.
        """
        output_dir = output_dir or os.path.join(self.train_args["output_dir"], "lora_adapter")
        os.makedirs(output_dir, exist_ok=True)

        # Save only the adapter
        self.model.save_pretrained(output_dir)

        print(f"LoRA adapter saved to {output_dir}")