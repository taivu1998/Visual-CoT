"""
Canonical dataset schema helpers and validation utilities.
"""
import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


BOX_PATTERN = re.compile(r"<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>")
REF_BOX_PATTERN = re.compile(r"<ref>([^<]+)</ref><box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>")


def load_jsonl_records(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def dump_jsonl_records(records: Iterable[Dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_boxes_from_text(text: str) -> List[List[int]]:
    return [[int(value) for value in match] for match in BOX_PATTERN.findall(text or "")]


def extract_refs_from_text(text: str) -> List[Tuple[str, List[int]]]:
    return [
        (match[0], [int(value) for value in match[1:]])
        for match in REF_BOX_PATTERN.findall(text or "")
    ]


def is_canonical_sample(sample: Dict[str, Any]) -> bool:
    return "image" in sample and "task" in sample and "answer" in sample


def is_legacy_messages_sample(sample: Dict[str, Any]) -> bool:
    return "messages" in sample


def _extract_user_image_and_question(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    image_path = None
    question = ""

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image" and image_path is None:
                    image_path = item.get("image")
                elif item.get("type") == "text":
                    question += item.get("text", "")
        elif isinstance(content, str):
            question += content

        if image_path or question:
            break

    return image_path, question.strip()


def legacy_messages_to_canonical(sample: Dict[str, Any], sample_id: Optional[str] = None) -> Dict[str, Any]:
    messages = sample.get("messages", [])
    image_path, question = _extract_user_image_and_question(messages)
    answer_text = ""
    for message in messages:
        if message.get("role") == "assistant":
            answer_text = message.get("content", "") or ""
            break

    refs = extract_refs_from_text(answer_text)
    structured_boxes = [
        {"label": label, "coords": coords, "normalized": True}
        for label, coords in refs
    ]

    canonical = {
        "id": sample_id or sample.get("id") or sample.get("metadata", {}).get("id"),
        "image": {
            "path": image_path,
            "width": sample.get("image", {}).get("width") if isinstance(sample.get("image"), dict) else None,
            "height": sample.get("image", {}).get("height") if isinstance(sample.get("image"), dict) else None,
        },
        "task": {
            "question": question,
        },
        "answer": {
            "text": answer_text,
            "boxes": structured_boxes,
        },
        "metadata": deepcopy(sample.get("metadata", {})),
    }

    if not canonical["id"]:
        canonical["id"] = None

    return canonical


def canonical_to_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_path = sample.get("image", {}).get("path")
    question = sample.get("task", {}).get("question", "")
    answer_text = sample.get("answer", {}).get("text", "")

    content: List[Dict[str, Any]] = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": question})

    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": answer_text},
    ]


def to_canonical_sample(sample: Dict[str, Any], sample_id: Optional[str] = None) -> Dict[str, Any]:
    if is_canonical_sample(sample):
        return deepcopy(sample)
    if is_legacy_messages_sample(sample):
        return legacy_messages_to_canonical(sample, sample_id=sample_id)
    raise ValueError("Unsupported sample format. Expected canonical schema or legacy 'messages' schema.")


def resolve_image_path(image_path: Optional[str], dataset_file: Optional[str] = None, repo_root: Optional[str] = None) -> Optional[str]:
    if not image_path:
        return None
    if os.path.isabs(image_path):
        return image_path

    roots = []
    if dataset_file:
        roots.append(os.path.dirname(os.path.abspath(dataset_file)))
    if repo_root:
        roots.append(os.path.abspath(repo_root))

    for root in roots:
        candidate = os.path.abspath(os.path.join(root, image_path))
        if os.path.exists(candidate):
            return candidate

    if roots:
        return os.path.abspath(os.path.join(roots[-1], image_path))
    return os.path.abspath(image_path)


def validate_canonical_sample(
    sample: Dict[str, Any],
    dataset_file: Optional[str] = None,
    repo_root: Optional[str] = None,
    require_image: bool = True,
    require_boxes: bool = False,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    sample_id = sample.get("id") or "<unknown>"
    image_path = sample.get("image", {}).get("path")
    question = sample.get("task", {}).get("question", "")
    answer_text = sample.get("answer", {}).get("text", "")
    structured_boxes = sample.get("answer", {}).get("boxes", [])

    if not image_path:
        if require_image:
            errors.append(f"{sample_id}: missing image path")
    else:
        resolved = resolve_image_path(image_path, dataset_file=dataset_file, repo_root=repo_root)
        if os.path.isabs(image_path):
            warnings.append(f"{sample_id}: image path is absolute and not portable: {image_path}")
        if require_image and (resolved is None or not os.path.exists(resolved)):
            errors.append(f"{sample_id}: image file does not exist: {image_path}")

    if not question:
        errors.append(f"{sample_id}: missing task.question")

    if not answer_text:
        errors.append(f"{sample_id}: missing answer.text")

    text_boxes = extract_boxes_from_text(answer_text)
    ref_boxes = extract_refs_from_text(answer_text)

    if "<ref>" in answer_text and not ref_boxes:
        errors.append(f"{sample_id}: answer contains <ref> tags without matching <box> tags")

    if answer_text.rstrip().endswith("</ref>"):
        errors.append(f"{sample_id}: answer appears truncated and ends with '</ref>'")

    if require_boxes and not structured_boxes and not text_boxes:
        errors.append(f"{sample_id}: sample requires grounding boxes but none were found")

    if structured_boxes and len(structured_boxes) != len(ref_boxes):
        warnings.append(
            f"{sample_id}: structured box count ({len(structured_boxes)}) does not match inline ref/box count ({len(ref_boxes)})"
        )

    all_boxes = structured_boxes or [{"coords": box} for box in text_boxes]
    for index, entry in enumerate(all_boxes):
        coords = entry.get("coords") if isinstance(entry, dict) else entry
        if not isinstance(coords, list) or len(coords) != 4:
            errors.append(f"{sample_id}: box {index} does not have four coordinates")
            continue
        x1, y1, x2, y2 = coords
        if x1 > x2 or y1 > y2:
            errors.append(f"{sample_id}: box {index} has inverted coordinates {coords}")
        for coordinate in coords:
            if coordinate < 0 or coordinate > 1000:
                warnings.append(f"{sample_id}: box {index} coordinate is outside normalized range 0-1000: {coords}")
                break

    return errors, warnings
