"""
CLI Inference & Evaluation.
Phase 4: Supports single inference and batch evaluation with IoU metrics.

Uses qwen_vl_utils for proper vision-language processing.
"""
import sys
import os
import re
import json
from typing import List, Tuple, Optional, Union
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastVisionModel
from PIL import Image
from src.utils import calculate_iou, parse_boxes_from_text, setup_logger

# Import qwen_vl_utils for proper vision processing
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not installed. Install with: pip install qwen-vl-utils")

logger = setup_logger("VCoT-Inference")


def load_model(model_path: str):
    """Load model for inference."""
    model, tokenizer = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    return model, tokenizer


def infer(
    model,
    tokenizer,
    image_input: Union[str, Image.Image],
    prompt: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.2
) -> str:
    """
    Run inference on a single image.

    Args:
        model: The loaded model
        tokenizer: The tokenizer/processor
        image_input: Either a path to image file or PIL Image
        prompt: The question/prompt to ask
        max_new_tokens: Maximum tokens to generate
        repetition_penalty: Penalty for repeated tokens

    Returns:
        Generated response text
    """
    # Load image if path provided
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB") if image_input.mode != "RGB" else image_input

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]

    # Use qwen_vl_utils if available for proper vision processing
    if QWEN_VL_UTILS_AVAILABLE:
        # Apply chat template to get the text prompt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info properly
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize with images
        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
    else:
        # Fallback to basic approach
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    if QWEN_VL_UTILS_AVAILABLE:
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def infer_with_url(
    model,
    tokenizer,
    image_url: str,
    prompt: str,
    max_new_tokens: int = 512
) -> str:
    """
    Run inference on an image from URL.

    Args:
        model: The loaded model
        tokenizer: The tokenizer/processor
        image_url: URL to the image
        prompt: The question/prompt to ask
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """
    import requests
    from io import BytesIO

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(image_url, headers=headers, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")

    return infer(model, tokenizer, image, prompt, max_new_tokens)


def evaluate_sample(
    model,
    tokenizer,
    image_path: str,
    question: str,
    gold_boxes: List[List[int]],
    gold_answer: Optional[str] = None
) -> dict:
    """
    Evaluate a single sample.
    Returns IoU scores for predicted vs gold boxes and answer accuracy.
    """
    response = infer(model, tokenizer, image_path, question)
    pred_boxes = parse_boxes_from_text(response)

    # Calculate IoU for each predicted box against gold boxes
    ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gold_box in gold_boxes:
            iou = calculate_iou(pred_box, gold_box)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)

    avg_iou = sum(ious) / len(ious) if ious else 0.0
    iou_success = sum(1 for iou in ious if iou > 0.5) / len(ious) if ious else 0.0

    result = {
        "response": response,
        "pred_boxes": pred_boxes,
        "avg_iou": avg_iou,
        "iou_success_rate": iou_success,
        "num_boxes_predicted": len(pred_boxes),
        "num_boxes_gold": len(gold_boxes),
    }

    return result


def evaluate_dataset(model, tokenizer, eval_jsonl_path: str) -> dict:
    """
    Evaluate model on a JSONL dataset.
    Each line should have: image_path, question, gold_boxes, gold_answer
    """
    results = []
    total_iou = 0.0
    total_success = 0.0

    with open(eval_jsonl_path, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Evaluating on {len(samples)} samples...")

    for i, sample in enumerate(samples):
        result = evaluate_sample(
            model, tokenizer,
            sample["image_path"],
            sample["question"],
            sample.get("gold_boxes", []),
            sample.get("gold_answer")
        )
        results.append(result)
        total_iou += result["avg_iou"]
        total_success += result["iou_success_rate"]

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)} samples")

    summary = {
        "num_samples": len(samples),
        "mean_iou": total_iou / len(samples) if samples else 0.0,
        "mean_iou_success_rate": total_success / len(samples) if samples else 0.0,
        "results": results,
    }

    return summary


def print_evaluation_table(base_results: dict, vcot_results: dict):
    """Print comparison table as mentioned in project plan."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<30} {'Base Model':<15} {'V-CoT':<15}")
    print("-"*60)
    print(f"{'Mean IoU':<30} {base_results['mean_iou']:<15.3f} {vcot_results['mean_iou']:<15.3f}")
    print(f"{'IoU Success Rate (>0.5)':<30} {base_results['mean_iou_success_rate']:<15.3f} {vcot_results['mean_iou_success_rate']:<15.3f}")
    print(f"{'Samples Evaluated':<30} {base_results['num_samples']:<15} {vcot_results['num_samples']:<15}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V-CoT Inference & Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--image_path", help="Path to single image for inference")
    parser.add_argument("--prompt", default="Explain the reasoning step by step.", help="Prompt for inference")
    parser.add_argument("--eval_jsonl", help="Path to evaluation JSONL for batch evaluation")
    parser.add_argument("--output_json", help="Path to save evaluation results")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    if args.eval_jsonl:
        # Batch evaluation mode
        results = evaluate_dataset(model, tokenizer, args.eval_jsonl)
        logger.info(f"Evaluation complete. Mean IoU: {results['mean_iou']:.3f}")

        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")
    else:
        # Single inference mode
        if not args.image_path:
            parser.error("--image_path is required for single inference mode")

        response = infer(model, tokenizer, args.image_path, args.prompt)
        print("\n" + "="*60)
        print("MODEL RESPONSE:")
        print("="*60)
        print(response)
        print("="*60 + "\n")