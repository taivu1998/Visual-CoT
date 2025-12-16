"""
Phase 5: The 'Wow' Factor Demo.
Implements TRUE STREAMING of bounding boxes as the text generates.

Usage:
    python app/app.py --model_path outputs/checkpoints
    python app/app.py --model_path unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit  # Use base model
"""
import gradio as gr
import re
import cv2
import numpy as np
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global model variables (loaded lazily)
model = None
tokenizer = None
MODEL_LOADED = False


def load_model(model_path: str):
    """Load the model with error handling."""
    global model, tokenizer, MODEL_LOADED

    if MODEL_LOADED:
        return True

    try:
        from unsloth import FastVisionModel

        print(f"Loading model from {model_path}...")

        if not os.path.exists(model_path) and not model_path.startswith("unsloth/"):
            print(f"Warning: Path '{model_path}' does not exist.")
            print("You can either:")
            print("  1. Train a model first: python scripts/train.py --config configs/default.yaml")
            print("  2. Use the base model: --model_path unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
            return False

        model, tokenizer = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
        FastVisionModel.for_inference(model)
        MODEL_LOADED = True
        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def stream_reasoning(image, question):
    """
    Stream reasoning with live bounding box visualization.

    Args:
        image: Input image as numpy array (RGB from Gradio)
        question: Question to ask about the image

    Yields:
        Tuple of (image with boxes, generated text)
    """
    global model, tokenizer

    if image is None:
        yield None, "Please upload an image."
        return

    if not MODEL_LOADED:
        yield image, "Model not loaded. Please restart the app with a valid model path."
        return

    from transformers import TextIteratorStreamer
    from threading import Thread

    # Setup Input - Gradio sends numpy array (RGB), convert for OpenCV
    orig_h, orig_w = image.shape[:2]
    overlay_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    # Setup Streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        decode_kwargs={"skip_special_tokens": False}
    )
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=512, use_cache=True)

    # Threaded Generation
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    generated_text = ""
    # Regex patterns for parsing
    box_pattern = re.compile(r"<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>")
    ref_pattern = re.compile(r"<ref>([^<]+)</ref><box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>")

    # Stream Loop
    for new_text in streamer:
        generated_text += new_text

        # Parse all refs with boxes found so far
        ref_matches = ref_pattern.findall(generated_text)

        # Redraw overlay
        current_overlay = overlay_img.copy()

        for match in ref_matches:
            label = match[0]
            x1, y1, x2, y2 = map(int, match[1:])

            # Denormalize from 0-1000 to pixel coordinates
            abs_x1 = int((x1 / 1000) * orig_w)
            abs_y1 = int((y1 / 1000) * orig_h)
            abs_x2 = int((x2 / 1000) * orig_w)
            abs_y2 = int((y2 / 1000) * orig_h)

            # Draw Green Box
            cv2.rectangle(current_overlay, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)

            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            label_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            cv2.rectangle(
                current_overlay,
                (abs_x1, abs_y1 - label_size[1] - 4),
                (abs_x1 + label_size[0] + 4, abs_y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                current_overlay, label,
                (abs_x1 + 2, abs_y1 - 2),
                font, font_scale, (0, 0, 0), 1
            )

        # Convert back to RGB for Gradio
        final_frame = cv2.cvtColor(current_overlay, cv2.COLOR_BGR2RGB)

        yield final_frame, generated_text

    thread.join()


def create_demo():
    """Create and return the Gradio demo interface."""
    with gr.Blocks(title="V-CoT Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # V-CoT: Grounded Visual Reasoning

        Upload an image and ask a question. The model will explain its reasoning
        step-by-step, highlighting objects it references with bounding boxes.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(label="Input Image", type="numpy")
                inp_txt = gr.Textbox(
                    label="Question",
                    value="Explain step-by-step what you see in this image.",
                    placeholder="Enter your question here..."
                )
                btn = gr.Button("Generate Reasoning", variant="primary")

            with gr.Column(scale=1):
                out_img = gr.Image(label="Live Visualization")
                out_txt = gr.Textbox(label="Reasoning Trace", lines=12)

        # Example inputs
        gr.Examples(
            examples=[
                ["Explain the reasoning step by step."],
                ["What objects can you identify? Point to each one."],
                ["Describe what is happening in this image."],
            ],
            inputs=[inp_txt],
        )

        btn.click(stream_reasoning, [inp_img, inp_txt], [out_img, out_txt])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-CoT Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/checkpoints",
        help="Path to trained model checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )

    args = parser.parse_args()

    # Load model
    if not load_model(args.model_path):
        print("\nFailed to load model. Starting demo in limited mode.")
        print("The demo will show an error message when you try to generate.\n")

    # Create and launch demo
    demo = create_demo()
    demo.launch(share=args.share, server_port=args.port)