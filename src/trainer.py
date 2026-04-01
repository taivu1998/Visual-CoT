"""
Training loop encapsulation for multimodal and text-only debug modes.
"""
import os
from typing import Optional

from datasets import Dataset
from PIL import Image
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments

from src.dataset import create_multimodal_dataset, extract_text_tokenizer, prepare_pretokenized_dataset
from src.runtime import get_process_vision_info


def _is_bfloat16_supported() -> bool:
    try:
        from unsloth import is_bfloat16_supported

        return is_bfloat16_supported()
    except Exception:
        return False


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [name for name in os.listdir(checkpoint_dir) if name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda value: int(value.split("-")[1]))
    return os.path.join(checkpoint_dir, latest)


class MultimodalDataCollator:
    """
    Collate canonical multimodal records into model-ready tensors.
    """

    def __init__(self, processor, ignore_index: int = -100):
        self.processor = processor
        self.ignore_index = ignore_index
        self.process_vision_info = get_process_vision_info()

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def __call__(self, features):
        batch_messages = [feature["messages"] for feature in features]
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch_messages
        ]

        if self.process_vision_info is not None:
            images = []
            videos = []
            for messages in batch_messages:
                sample_images, sample_videos = self.process_vision_info(messages)
                images.append(sample_images[0] if isinstance(sample_images, list) and sample_images else sample_images)
                videos.append(sample_videos[0] if isinstance(sample_videos, list) and sample_videos else None)

            if any(video is not None for video in videos):
                batch = self.processor(
                    text=texts,
                    images=images,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                batch = self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
        else:
            images = [self._load_image(feature["image_path"]) for feature in features]
            batch = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

        labels = batch["input_ids"].clone()
        pad_token_id = getattr(self.processor, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.processor, "tokenizer"):
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = self.ignore_index
        batch["labels"] = labels
        return batch


class VCoTTrainer:
    """
    Visual Chain-of-Thought trainer with two explicit modes:

    - multimodal: default and recommended
    - text_only_debug: lightweight smoke-testing path
    """

    def __init__(
        self,
        model,
        processor,
        train_dataset: Dataset,
        config: dict,
        val_dataset: Optional[Dataset] = None,
        mode: str = "multimodal",
        text_tokenizer=None,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.train_args = config["training"]
        self.mode = mode
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.text_tokenizer = text_tokenizer or extract_text_tokenizer(processor)

    @classmethod
    def from_files(
        cls,
        model,
        processor,
        train_file: str,
        config: dict,
        val_file: Optional[str] = None,
        mode: str = "multimodal",
        repo_root: Optional[str] = None,
    ) -> "VCoTTrainer":
        max_seq_length = config["model"]["max_seq_length"]

        if mode == "text_only_debug":
            train_dataset, text_tokenizer = prepare_pretokenized_dataset(train_file, processor, max_seq_length)
            val_dataset = None
            if val_file and os.path.exists(val_file):
                val_dataset, _ = prepare_pretokenized_dataset(val_file, processor, max_seq_length)
            return cls(
                model=model,
                processor=processor,
                train_dataset=train_dataset,
                config=config,
                val_dataset=val_dataset,
                mode=mode,
                text_tokenizer=text_tokenizer,
            )

        train_dataset = create_multimodal_dataset(train_file, processor, repo_root=repo_root)
        val_dataset = None
        if val_file and os.path.exists(val_file):
            val_dataset = create_multimodal_dataset(val_file, processor, repo_root=repo_root)

        return cls(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            mode=mode,
        )

    def _build_training_arguments(self, output_dir: str) -> TrainingArguments:
        batch_size = self.train_args["per_device_train_batch_size"]
        grad_accum = self.train_args["gradient_accumulation_steps"]
        total_samples = len(self.train_dataset)
        effective_batch = max(1, batch_size * grad_accum)
        steps_per_epoch = max(1, total_samples // effective_batch)

        num_epochs = self.train_args.get("num_epochs")
        max_steps = self.train_args.get("max_steps", -1)
        eval_enabled = self.val_dataset is not None

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs if max_steps == -1 else None,
            max_steps=max_steps if max_steps != -1 else -1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=float(self.train_args["learning_rate"]),
            lr_scheduler_type=self.train_args.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.train_args.get("warmup_ratio", 0.03),
            warmup_steps=self.train_args.get("warmup_steps", 0),
            weight_decay=self.train_args.get("weight_decay", 0.01),
            max_grad_norm=self.train_args.get("max_grad_norm", 0.3),
            fp16=not _is_bfloat16_supported(),
            bf16=_is_bfloat16_supported(),
            gradient_checkpointing=self.train_args.get("gradient_checkpointing", True),
            optim=self.train_args.get("optimizer", "adamw_torch"),
            evaluation_strategy="steps" if eval_enabled else "no",
            eval_steps=self.train_args.get("eval_steps", max(10, steps_per_epoch)),
            load_best_model_at_end=eval_enabled,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_strategy="steps",
            save_steps=self.train_args.get("save_steps", max(10, steps_per_epoch)),
            save_total_limit=self.train_args.get("save_total_limit", 3),
            logging_steps=self.train_args.get("logging_steps", 50),
            logging_first_step=True,
            report_to=self.config["project"].get("report_to", "none"),
            seed=self.config["project"].get("seed", 42),
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

    def _build_data_collator(self):
        if self.mode == "text_only_debug":
            return DataCollatorForSeq2Seq(
                tokenizer=self.text_tokenizer,
                padding=True,
                return_tensors="pt",
            )
        return MultimodalDataCollator(self.processor)

    def train(self, resume_from_checkpoint: Optional[str] = None):
        output_dir = self.train_args["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        if resume_from_checkpoint is True:
            resume_from_checkpoint = find_latest_checkpoint(output_dir)
            if resume_from_checkpoint:
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            else:
                print("No checkpoint found, starting fresh training")
                resume_from_checkpoint = None

        args = self._build_training_arguments(output_dir)

        callbacks = []
        if self.val_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_args.get("early_stopping_patience", 3),
                    early_stopping_threshold=self.train_args.get("early_stopping_threshold", 0.001),
                )
            )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self._build_data_collator(),
            tokenizer=self.text_tokenizer if self.mode == "text_only_debug" else None,
            callbacks=callbacks or None,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self._save_checkpoint()
        return trainer

    def _save_checkpoint(self):
        from src.config_loader import dump_config

        output_dir = self.train_args["output_dir"]
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        dump_config(self.config, os.path.join(output_dir, "training_config.yaml"))
        print(f"Checkpoint saved to {output_dir}")

    def save_lora_only(self, output_dir: Optional[str] = None):
        output_dir = output_dir or os.path.join(self.train_args["output_dir"], "lora_adapter")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        print(f"LoRA adapter saved to {output_dir}")
