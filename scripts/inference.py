"""
CLI inference and evaluation.

Supports:
- single-image inference
- batch evaluation on canonical or legacy JSONL data
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_schema import resolve_image_path, to_canonical_sample
from src.generation import generate_response
from src.model import load_model_for_inference
from src.utils import calculate_iou, parse_boxes_from_text, setup_logger

logger = setup_logger("VCoT-Inference")


def load_model(model_path: str):
    model, processor, backend = load_model_for_inference(model_path)
    logger.info("Loaded inference model using backend: %s", backend)
    return model, processor


def infer(
    model,
    processor,
    image_input,
    prompt: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.2,
) -> str:
    return generate_response(
        model=model,
        processor=processor,
        image_input=image_input,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )


def _evaluation_sample_from_record(record: Dict, dataset_file: str) -> Dict:
    canonical = to_canonical_sample(record)
    boxes = []
    for entry in canonical.get("answer", {}).get("boxes", []):
        coords = entry.get("coords")
        if coords:
            boxes.append(coords)
    if not boxes:
        boxes = parse_boxes_from_text(canonical.get("answer", {}).get("text", ""))

    return {
        "id": canonical.get("id"),
        "image_path": resolve_image_path(
            canonical.get("image", {}).get("path"),
            dataset_file=dataset_file,
            repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ),
        "question": canonical.get("task", {}).get("question", ""),
        "gold_boxes": boxes,
        "gold_answer": canonical.get("answer", {}).get("text", ""),
    }


def evaluate_sample(
    model,
    processor,
    image_path: str,
    question: str,
    gold_boxes: List[List[int]],
    gold_answer: Optional[str] = None,
) -> dict:
    response = infer(model, processor, image_path, question)
    pred_boxes = parse_boxes_from_text(response)

    ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gold_box in gold_boxes:
            iou = calculate_iou(pred_box, gold_box)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)

    answer_match = False
    if gold_answer:
        answer_match = gold_answer.strip() in response

    return {
        "response": response,
        "pred_boxes": pred_boxes,
        "avg_iou": sum(ious) / len(ious) if ious else 0.0,
        "iou_success_rate": (sum(1 for iou in ious if iou > 0.5) / len(ious)) if ious else 0.0,
        "num_boxes_predicted": len(pred_boxes),
        "num_boxes_gold": len(gold_boxes),
        "answer_match": answer_match,
    }


def evaluate_dataset(model, processor, eval_jsonl_path: str) -> dict:
    with open(eval_jsonl_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    samples = [
        _evaluation_sample_from_record(record, dataset_file=eval_jsonl_path)
        for record in records
    ]
    logger.info("Evaluating on %s samples...", len(samples))

    results = []
    total_iou = 0.0
    total_success = 0.0
    total_answer_match = 0

    for index, sample in enumerate(samples):
        result = evaluate_sample(
            model=model,
            processor=processor,
            image_path=sample["image_path"],
            question=sample["question"],
            gold_boxes=sample["gold_boxes"],
            gold_answer=sample.get("gold_answer"),
        )
        results.append(
            {
                "id": sample.get("id"),
                **result,
            }
        )
        total_iou += result["avg_iou"]
        total_success += result["iou_success_rate"]
        total_answer_match += 1 if result["answer_match"] else 0

        if (index + 1) % 10 == 0:
            logger.info("Processed %s/%s samples", index + 1, len(samples))

    count = len(samples) or 1
    return {
        "num_samples": len(samples),
        "mean_iou": total_iou / count,
        "mean_iou_success_rate": total_success / count,
        "answer_match_rate": total_answer_match / count,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="V-CoT Inference & Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint or model ID")
    parser.add_argument("--image_path", help="Path to single image for inference")
    parser.add_argument("--prompt", default="Explain the reasoning step by step.", help="Prompt for inference")
    parser.add_argument("--eval_jsonl", help="Path to evaluation JSONL for batch evaluation")
    parser.add_argument("--output_json", help="Path to save evaluation results")

    args = parser.parse_args()
    model, processor = load_model(args.model_path)

    if args.eval_jsonl:
        results = evaluate_dataset(model, processor, args.eval_jsonl)
        logger.info("Evaluation complete. Mean IoU: %.3f", results["mean_iou"])
        if args.output_json:
            output_dir = os.path.dirname(args.output_json)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
            logger.info("Results saved to %s", args.output_json)
        return

    if not args.image_path:
        parser.error("--image_path is required for single inference mode")

    response = infer(model, processor, args.image_path, args.prompt)
    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
