"""
Dataset Logic for Unsloth/Qwen.

Handles loading and formatting data for vision-language model training.
Includes optimized pre-tokenization approach for VLM processors.
"""
import json
import os
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset
from PIL import Image
from tqdm import tqdm


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    """
    Loads JSONL data.

    Format expected by Unsloth Qwen2.5:
    {
      "messages": [
        {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": "Trace..."}
      ]
    }
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return Dataset.from_list(data)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file as list of dicts."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_text_tokenizer(tokenizer):
    """
    Extract the underlying text tokenizer from a VLM processor.

    Qwen2-VL and similar models use a processor that wraps both image
    and text tokenizers. This function extracts just the text tokenizer.
    """
    # Check for nested tokenizer (common in VLM processors)
    if hasattr(tokenizer, 'tokenizer'):
        return tokenizer.tokenizer
    elif hasattr(tokenizer, 'text_tokenizer'):
        return tokenizer.text_tokenizer
    else:
        # Try to load a separate text tokenizer
        try:
            from transformers import AutoTokenizer
            # Fallback to Qwen2.5-7B-Instruct text tokenizer
            return AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                trust_remote_code=True
            )
        except Exception:
            # Return the original tokenizer as fallback
            return tokenizer


def convert_messages_to_text(messages: List[Dict]) -> str:
    """
    Convert messages to plain text format without using chat template.

    This avoids issues with the VLM tokenizer's special handling of
    multimodal content. Uses Qwen chat format with im_start/im_end tokens.
    """
    text_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            # Handle user content which may be a list (with image + text)
            if isinstance(content, list):
                # Extract just the text parts, skip image references
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    # Add system message if not present
    if not any("<|im_start|>system" in p for p in text_parts):
        text_parts.insert(0, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>")

    return "\n".join(text_parts)


def prepare_text_only_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 2048
) -> List[Dict]:
    """
    Prepare dataset by pre-tokenizing to avoid VLM collator issues.

    This approach tokenizes the text portion separately, which is more
    reliable than using the full VLM processor during training.

    Args:
        data: List of samples with "messages" field
        tokenizer: Text tokenizer (use extract_text_tokenizer first)
        max_length: Maximum sequence length

    Returns:
        List of tokenized samples with input_ids, attention_mask, labels
    """
    processed = []
    skipped = 0
    first_error = None

    for i, sample in enumerate(tqdm(data, desc="Tokenizing")):
        try:
            messages = sample.get("messages", [])

            # Convert to plain text format
            text = convert_messages_to_text(messages)

            if not text or len(text) < 10:
                skipped += 1
                continue

            # Tokenize with the TEXT tokenizer (not the VLM processor)
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            # Add labels (same as input_ids for causal LM)
            processed.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": encoding["input_ids"].copy(),
            })

        except Exception as e:
            if first_error is None:
                first_error = f"Sample {i}: {str(e)[:200]}"
            skipped += 1
            continue

    if skipped > 0:
        print(f"  Warning: Skipped {skipped} samples")
        if first_error:
            print(f"  First error: {first_error}")

    return processed


def format_for_unsloth(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Format a single sample for Unsloth VLM training.

    Converts the messages format to the text format expected by the trainer.
    For vision models, Unsloth handles the multimodal inputs directly.
    """
    messages = sample.get("messages", [])

    # Apply chat template to convert messages to text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text, "messages": messages}


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int = 2048
) -> Dataset:
    """
    Prepare dataset for training by formatting samples.
    """
    def process_sample(sample):
        return format_for_unsloth(sample, tokenizer)

    return dataset.map(process_sample, num_proc=1)


def prepare_pretokenized_dataset(
    file_path: str,
    tokenizer,
    max_seq_length: int = 2048
) -> Dataset:
    """
    Load and pre-tokenize a JSONL dataset.

    This is the recommended approach for VLM training as it avoids
    issues with the multimodal processor during training.

    Args:
        file_path: Path to JSONL file
        tokenizer: VLM processor/tokenizer (text tokenizer will be extracted)
        max_seq_length: Maximum sequence length

    Returns:
        HuggingFace Dataset with tokenized samples
    """
    # Extract text tokenizer from VLM processor
    text_tokenizer = extract_text_tokenizer(tokenizer)

    # Load raw data
    raw_data = load_jsonl(file_path)
    print(f"Loaded {len(raw_data)} samples from {file_path}")

    # Pre-tokenize
    tokenized_data = prepare_text_only_dataset(raw_data, text_tokenizer, max_seq_length)
    print(f"Tokenized {len(tokenized_data)} samples")

    return Dataset.from_list(tokenized_data), text_tokenizer


def load_image_from_path(image_path: str) -> Optional[Image.Image]:
    """Load an image from path, handling errors gracefully."""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        return None
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def create_sample_dataset() -> Dataset:
    """
    Create a small sample dataset for testing.
    Useful for verifying the training pipeline works.
    """
    samples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "I can see various objects in the image. Let me analyze them step by step."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Explain what is happening."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "Looking at the <ref>main subject</ref><box>[100, 100, 500, 500]</box>, I can explain the scene."
                }
            ]
        }
    ]
    return Dataset.from_list(samples)