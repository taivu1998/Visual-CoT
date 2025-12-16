"""
Phase 2: Synthetic Data Generation.

This script supports two data sources:
1. ScienceQA dataset with GPT-4o generated reasoning traces
2. VisCOT dataset (deepcs233/Visual-CoT) with real bounding boxes (recommended)

Usage:
    # Use VisCOT dataset (recommended - 150K samples with real bboxes)
    python scripts/generate_data.py --source viscot --output_dir data --max_samples 150000

    # Use ScienceQA with GPT-4o distillation
    python scripts/generate_data.py --source scienceqa --output_dir data --max_samples 2000

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (only needed for scienceqa source)
"""
import os
import sys
import json
import base64
import asyncio
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

# Full system prompt as specified in the project plan
SYSTEM_PROMPT = """
You are an expert Visual Reasoning Assistant. Your goal is to explain the answer to a science question step-by-step.
CRITICAL RULE: Whenever you mention a physical object in the image that supports your reasoning, you MUST immediately follow it with its bounding box in the format: <ref>object_name</ref><box>[x_min, y_min, x_max, y_max]</box>.
- Coordinates must be normalized from 0 to 1000.
- (0,0) is top-left, (1000,1000) is bottom-right.
- Example: "The <ref>red gear</ref><box>[100, 200, 300, 400]</box> turns clockwise."
"""

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS = 10
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5


# =============================================================================
# VisCOT Dataset Conversion (Recommended)
# =============================================================================

def parse_bbox_from_text(text: str) -> Optional[List[int]]:
    """
    Extract bbox from text like '[0.562, 0.228, 0.646, 0.292]'.

    Returns normalized coordinates in 0-1000 range.
    """
    match = re.search(r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', text)
    if match:
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            # Check if coordinates are in 0-1 range (normalized)
            if all(0 <= c <= 1 for c in coords):
                return [int(c * 1000) for c in coords]
            # Or if they're already in 0-1000 range
            elif all(c <= 1000 for c in coords):
                return [int(c) for c in coords]
        except (ValueError, TypeError):
            pass
    return None


def format_bbox(bbox: List[int], fmt: str = "vcot") -> Optional[str]:
    """
    Format bounding box coordinates.

    Args:
        bbox: Box coordinates [x1, y1, x2, y2] in 0-1000 range
        fmt: Format type - "vcot" for <box>[x1, y1, x2, y2]</box>
             or "qwen_native" for Qwen's native format

    Returns:
        Formatted bbox string
    """
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    if fmt == "vcot":
        return f"<box>[{x1}, {y1}, {x2}, {y2}]</box>"
    else:  # qwen_native
        return f"<|box_start|>({y1},{x1}),({y2},{x2})<|box_end|>"


def convert_viscot_sample(sample: Dict, bbox_format: str = "vcot", filter_quality: bool = True) -> Optional[Dict]:
    """
    Convert a single VisCOT sample to V-CoT training format.

    Args:
        sample: Raw sample from VisCOT dataset
        bbox_format: "vcot" or "qwen_native"
        filter_quality: If True, only keep samples with valid bboxes

    Returns:
        Converted sample or None if invalid
    """
    conversations = sample.get('conversations', [])
    images = sample.get('image', [])

    if not conversations or len(conversations) < 2:
        return None

    # Extract image path
    image_path = None
    for img in images:
        if isinstance(img, str) and '###' not in img:
            image_path = img
            break
    if not image_path and images:
        image_path = images[0].split('###')[0] if '###' in str(images[0]) else images[0]

    if not image_path:
        return None

    # Parse conversation
    question = None
    bbox = None
    answer = None

    for turn in conversations:
        role = turn.get('from', '')
        value = turn.get('value', '')

        if role == 'human' and '<image>' in value:
            q = value.replace('<image>', '').strip()
            if q and not question:
                question = q
        elif role == 'gpt':
            parsed_bbox = parse_bbox_from_text(value)
            if parsed_bbox and not bbox:
                bbox = parsed_bbox
            elif value and not value.startswith('['):
                answer = value.strip()

    if not question:
        return None

    # Build response
    response_parts = []
    if answer:
        response_parts.append(answer)

    has_bbox = False
    if bbox:
        bbox_str = format_bbox(bbox, bbox_format)
        if bbox_str:
            has_bbox = True
            obj_ref = answer[:30] if answer and len(answer) < 50 else "relevant region"
            response_parts.append(f"\n\nThe <ref>{obj_ref}</ref>{bbox_str} shows this in the image.")

    if not response_parts:
        return None

    # Quality filter: require bbox if enabled
    if filter_quality and not has_bbox:
        return None

    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": "".join(response_parts)}
        ],
        "metadata": {
            "source": "viscot",
            "dataset": sample.get('dataset', 'unknown'),
            "has_bbox": has_bbox
        }
    }


def load_and_convert_viscot(
    max_samples: int = 150000,
    bbox_format: str = "vcot",
    filter_quality: bool = True
) -> List[Dict]:
    """
    Load and convert the VisCOT dataset.

    Args:
        max_samples: Maximum number of samples to convert
        bbox_format: "vcot" or "qwen_native"
        filter_quality: If True, only keep samples with valid bboxes

    Returns:
        List of converted samples
    """
    print(f"Loading VisCOT dataset from HuggingFace...")
    dataset = load_dataset("deepcs233/Visual-CoT", data_files="viscot_363k.json", split="train")
    print(f"Loaded {len(dataset)} samples from VisCOT")

    print(f"\nConverting up to {max_samples} samples...")
    converted = []
    with_bbox = 0

    for i, sample in enumerate(tqdm(dataset, total=min(len(dataset), max_samples))):
        if len(converted) >= max_samples:
            break
        result = convert_viscot_sample(sample, bbox_format, filter_quality)
        if result:
            converted.append(result)
            if result["metadata"]["has_bbox"]:
                with_bbox += 1

    print(f"\nConverted: {len(converted)} samples")
    print(f"With bounding boxes: {with_bbox} ({100*with_bbox/max(1,len(converted)):.1f}%)")

    return converted


def analyze_converted_data(file_path: str) -> Dict:
    """
    Analyze converted dataset for valid bounding boxes and quality.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total': 0,
        'has_vcot_box': 0,
        'has_qwen_box': 0,
        'has_ref': 0,
        'avg_response_len': 0
    }

    response_lengths = []

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            stats['total'] += 1
            sample = json.loads(line)
            response = sample['messages'][1]['content']

            response_lengths.append(len(response))

            if '<ref>' in response:
                stats['has_ref'] += 1
            if '<box>[' in response:
                stats['has_vcot_box'] += 1
            if '<|box_start|>' in response:
                stats['has_qwen_box'] += 1

    stats['avg_response_len'] = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    return stats


# =============================================================================
# ScienceQA Dataset with GPT-4o (Alternative)
# =============================================================================

def get_image_mime_type(image_path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    return mime_types.get(ext, 'image/jpeg')


async def process_single_image(
    client: AsyncOpenAI,
    image_data: bytes,
    question: str,
    answer: str,
    semaphore: asyncio.Semaphore,
    mime_type: str = "image/jpeg"
) -> Optional[str]:
    """
    Process a single image with GPT-4o.
    Uses semaphore for rate limiting.
    """
    async with semaphore:
        b64 = base64.b64encode(image_data).decode('utf-8')

        # Include the answer in the prompt to guide the reasoning
        user_prompt = f"""Question: {question}

The correct answer is: {answer}

Please explain step-by-step how to arrive at this answer by carefully examining the image.
Remember to annotate every object you mention with its bounding box."""

        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:{mime_type};base64,{b64}",
                                "detail": "high"
                            }}
                        ]}
                    ],
                    max_tokens=1024,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"Failed after {RETRY_ATTEMPTS} attempts: {e}")
                    return None


async def process_batch(
    client: AsyncOpenAI,
    samples: List[Dict],
    semaphore: asyncio.Semaphore
) -> List[Dict]:
    """Process a batch of samples concurrently."""
    tasks = []
    for sample in samples:
        task = process_single_image(
            client,
            sample["image_bytes"],
            sample["question"],
            sample["answer"],
            semaphore,
            sample.get("mime_type", "image/jpeg")
        )
        tasks.append(task)

    results = await tqdm_asyncio.gather(*tasks, desc="Processing batch")

    processed = []
    for sample, response in zip(samples, results):
        if response:
            processed.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": sample["image_path"]},
                            {"type": "text", "text": sample["question"]}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "metadata": {
                    "source": "scienceqa",
                    "original_answer": sample["answer"],
                    "subject": sample.get("subject", ""),
                    "topic": sample.get("topic", ""),
                }
            })

    return processed


def load_scienceqa_images(max_samples: int = 2000, split: str = "train") -> List[Dict]:
    """
    Load ScienceQA dataset, filtering for samples with images.
    """
    print(f"Loading ScienceQA dataset ({split} split)...")
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)

    samples = []
    for item in dataset:
        # Only include samples that have images
        if item.get("image") is not None:
            # Get the correct answer text
            choices = item.get("choices", [])
            answer_idx = item.get("answer", 0)
            answer_text = choices[answer_idx] if answer_idx < len(choices) else ""

            # Convert PIL Image to bytes
            from io import BytesIO
            img_buffer = BytesIO()
            item["image"].save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            samples.append({
                "image_bytes": img_bytes,
                "image_path": f"scienceqa_{len(samples)}.png",  # Placeholder path
                "question": item.get("question", ""),
                "answer": answer_text,
                "subject": item.get("subject", ""),
                "topic": item.get("topic", ""),
                "mime_type": "image/png",
            })

            if len(samples) >= max_samples:
                break

    print(f"Loaded {len(samples)} samples with images")
    return samples


def save_images_to_disk(samples: List[Dict], output_dir: Path) -> List[Dict]:
    """Save images to disk and update paths."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    updated_samples = []
    for i, sample in enumerate(samples):
        image_filename = f"sample_{i:05d}.png"
        image_path = images_dir / image_filename

        with open(image_path, "wb") as f:
            f.write(sample["image_bytes"])

        sample_copy = sample.copy()
        sample_copy["image_path"] = str(image_path)
        updated_samples.append(sample_copy)

    return updated_samples


async def main_scienceqa(args):
    """Generate data using ScienceQA + GPT-4o."""
    from openai import AsyncOpenAI

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Load dataset
    samples = load_scienceqa_images(max_samples=args.max_samples)

    if not samples:
        print("No samples found! Check dataset availability.")
        sys.exit(1)

    # Optionally save images to disk
    if args.save_images:
        print("Saving images to disk...")
        samples = save_images_to_disk(samples, output_dir)

    # Process in batches
    all_processed = []
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1}/{(len(samples) - 1) // args.batch_size + 1}")
        processed = await process_batch(client, batch, semaphore)
        all_processed.extend(processed)

        # Save intermediate results
        intermediate_path = output_dir / "intermediate.jsonl"
        with open(intermediate_path, "w") as f:
            for item in all_processed:
                f.write(json.dumps(item) + "\n")

    return all_processed


def main_viscot(args):
    """Generate data using VisCOT dataset (recommended)."""
    # Load and convert VisCOT dataset
    all_processed = load_and_convert_viscot(
        max_samples=args.max_samples,
        bbox_format=args.bbox_format,
        filter_quality=args.filter_quality
    )

    if not all_processed:
        print("No samples converted! Check dataset availability.")
        sys.exit(1)

    return all_processed


def main():
    parser = argparse.ArgumentParser(description="Generate V-CoT training data")
    parser.add_argument("--source", type=str, default="viscot",
                        choices=["viscot", "scienceqa"],
                        help="Data source: viscot (recommended) or scienceqa")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for processed data")
    parser.add_argument("--max_samples", type=int, default=150000,
                        help="Maximum number of samples to process")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")

    # VisCOT-specific arguments
    parser.add_argument("--bbox_format", type=str, default="vcot",
                        choices=["vcot", "qwen_native"],
                        help="Bounding box format (viscot source only)")
    parser.add_argument("--filter_quality", action="store_true", default=True,
                        help="Only keep samples with valid bboxes (viscot source only)")
    parser.add_argument("--no_filter_quality", action="store_false", dest="filter_quality",
                        help="Keep all samples regardless of bbox validity")

    # ScienceQA-specific arguments
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for GPT-4o processing (scienceqa source only)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save images to disk (scienceqa source only)")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data based on source
    print(f"{'='*60}")
    print(f"V-CoT DATA GENERATION")
    print(f"{'='*60}")
    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}\n")

    if args.source == "viscot":
        all_processed = main_viscot(args)
    else:
        all_processed = asyncio.run(main_scienceqa(args))

    if not all_processed:
        print("No data generated!")
        sys.exit(1)

    # Split into train/val
    val_size = max(1, int(len(all_processed) * args.val_split))
    val_data = all_processed[:val_size]
    train_data = all_processed[val_size:]

    # Save final datasets
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n{'='*60}")
    print("DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {len(all_processed)}")
    print(f"Training samples: {len(train_data)} -> {train_path}")
    print(f"Validation samples: {len(val_data)} -> {val_path}")

    # Analyze the generated data
    print(f"\n--- Data Analysis ---")
    train_stats = analyze_converted_data(train_path)
    print(f"Training data:")
    print(f"  Total samples: {train_stats['total']}")
    print(f"  With <ref> tags: {train_stats['has_ref']} ({100*train_stats['has_ref']/max(1,train_stats['total']):.1f}%)")
    print(f"  With V-CoT <box>: {train_stats['has_vcot_box']} ({100*train_stats['has_vcot_box']/max(1,train_stats['total']):.1f}%)")
    print(f"  Avg response length: {train_stats['avg_response_len']:.0f} chars")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()