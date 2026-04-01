"""
Shared multimodal preprocessing and generation utilities.
"""
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.runtime import get_process_vision_info


ImageInput = Union[str, Image.Image, np.ndarray]


def load_image(image_input: ImageInput) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        return Image.fromarray(image_input.astype("uint8")).convert("RGB")

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        return Image.open(image_input).convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image_input)!r}")


def build_messages(image_input: ImageInput, prompt: str) -> List[Dict[str, Any]]:
    image = load_image(image_input)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_generation_inputs(processor, image_input: ImageInput, prompt: str, model=None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    messages = build_messages(image_input, prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    process_vision_info = get_process_vision_info()
    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        image = messages[0]["content"][0]["image"]
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

    device = _get_model_device(model) if model is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    return inputs, messages


def decode_generation(processor, inputs: Dict[str, Any], outputs) -> str:
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated_ids = outputs[:, input_ids.shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


def generate_response(
    model,
    processor,
    image_input: ImageInput,
    prompt: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.2,
    do_sample: bool = False,
) -> str:
    inputs, _ = prepare_generation_inputs(processor, image_input, prompt, model=model)

    eos_token_id = getattr(processor, "eos_token_id", None)
    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        )

    return decode_generation(processor, inputs, outputs)
