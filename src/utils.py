"""
Utilities for logging, reproducibility, metrics, and visualization.
"""
import os
import re
import random
import logging
from typing import List, Tuple, Optional
import torch
import numpy as np
import cv2
from PIL import Image


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Setup a logger with console and file handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Compute IoU (Intersection over Union) between two boxes.

    Args:
        boxA: First box as [x1, y1, x2, y2]
        boxB: Second box as [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def parse_boxes_from_text(text: str) -> List[List[int]]:
    """
    Extract bounding boxes from model output text.

    Parses the format: <box>[x1, y1, x2, y2]</box>
    Coordinates are normalized 0-1000.

    Args:
        text: Model output text containing box annotations

    Returns:
        List of boxes, each as [x1, y1, x2, y2]
    """
    pattern = r"<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>"
    matches = re.findall(pattern, text)
    return [[int(x) for x in match] for match in matches]


def parse_refs_from_text(text: str) -> List[Tuple[str, List[int]]]:
    """
    Extract object references with their bounding boxes from model output.

    Parses the format: <ref>object_name</ref><box>[x1, y1, x2, y2]</box>

    Args:
        text: Model output text containing ref and box annotations

    Returns:
        List of (object_name, box) tuples
    """
    pattern = r"<ref>([^<]+)</ref><box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>"
    matches = re.findall(pattern, text)
    return [(match[0], [int(x) for x in match[1:]]) for match in matches]


def denormalize_box(
    box: List[int],
    img_width: int,
    img_height: int,
    coord_max: int = 1000
) -> List[int]:
    """
    Convert normalized coordinates (0-1000) to pixel coordinates.

    Args:
        box: Normalized box [x1, y1, x2, y2] in range 0-coord_max
        img_width: Image width in pixels
        img_height: Image height in pixels
        coord_max: Maximum coordinate value (default 1000)

    Returns:
        Box in pixel coordinates
    """
    x1, y1, x2, y2 = box
    return [
        int((x1 / coord_max) * img_width),
        int((y1 / coord_max) * img_height),
        int((x2 / coord_max) * img_width),
        int((y2 / coord_max) * img_height)
    ]


def normalize_box(
    box: List[int],
    img_width: int,
    img_height: int,
    coord_max: int = 1000
) -> List[int]:
    """
    Convert pixel coordinates to normalized coordinates (0-1000).

    Args:
        box: Pixel box [x1, y1, x2, y2]
        img_width: Image width in pixels
        img_height: Image height in pixels
        coord_max: Maximum coordinate value (default 1000)

    Returns:
        Normalized box
    """
    x1, y1, x2, y2 = box
    return [
        int((x1 / img_width) * coord_max),
        int((y1 / img_height) * coord_max),
        int((x2 / img_width) * coord_max),
        int((y2 / img_height) * coord_max)
    ]


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: List[List[int]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    normalized: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: Image as numpy array (BGR or RGB)
        boxes: List of boxes, each as [x1, y1, x2, y2]
        labels: Optional labels for each box
        color: Box color as (B, G, R)
        thickness: Line thickness
        normalized: If True, boxes are in 0-1000 range and need denormalization

    Returns:
        Image with boxes drawn
    """
    overlay = image.copy()
    img_h, img_w = image.shape[:2]

    for i, box in enumerate(boxes):
        if normalized:
            box = denormalize_box(box, img_w, img_h)

        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        if labels and i < len(labels):
            label = labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            label_size = cv2.getTextSize(label, font, font_scale, 1)[0]

            # Draw label background
            cv2.rectangle(
                overlay,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0] + 4, y1),
                color,
                -1
            )
            # Draw label text
            cv2.putText(
                overlay, label, (x1 + 2, y1 - 2),
                font, font_scale, (255, 255, 255), 1
            )

    return overlay


def visualize_reasoning_trace(
    image_path: str,
    reasoning_text: str,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize a reasoning trace by drawing all referenced objects on the image.

    Args:
        image_path: Path to the input image
        reasoning_text: Model output containing <ref> and <box> tags
        output_path: Optional path to save the visualization

    Returns:
        Image with visualization
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    refs = parse_refs_from_text(reasoning_text)
    labels = [ref[0] for ref in refs]
    boxes = [ref[1] for ref in refs]

    result = draw_boxes_on_image(image, boxes, labels, normalized=True)

    if output_path:
        cv2.imwrite(output_path, result)

    return result


def calculate_batch_iou(
    pred_boxes: List[List[int]],
    gold_boxes: List[List[int]],
    threshold: float = 0.5
) -> dict:
    """
    Calculate IoU metrics for a batch of predictions vs gold boxes.

    Args:
        pred_boxes: List of predicted boxes
        gold_boxes: List of gold/ground-truth boxes
        threshold: IoU threshold for considering a match successful

    Returns:
        Dictionary with metrics: mean_iou, success_rate, matched_count
    """
    if not pred_boxes or not gold_boxes:
        return {
            "mean_iou": 0.0,
            "success_rate": 0.0,
            "matched_count": 0,
            "pred_count": len(pred_boxes),
            "gold_count": len(gold_boxes)
        }

    ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gold_box in gold_boxes:
            iou = calculate_iou(pred_box, gold_box)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)

    mean_iou = sum(ious) / len(ious)
    success_count = sum(1 for iou in ious if iou > threshold)

    return {
        "mean_iou": mean_iou,
        "success_rate": success_count / len(ious),
        "matched_count": success_count,
        "pred_count": len(pred_boxes),
        "gold_count": len(gold_boxes)
    }