"""
Gradio demo with streaming bounding-box visualization.
"""
import argparse
import os
import re
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import prepare_generation_inputs
from src.model import load_model_for_inference

model = None
processor = None
MODEL_LOADED = False
MODEL_ERROR = None


def load_model(model_path: str):
    global model, processor, MODEL_LOADED, MODEL_ERROR

    if MODEL_LOADED:
        return True

    try:
        model, processor, backend = load_model_for_inference(model_path)
        MODEL_LOADED = True
        MODEL_ERROR = None
        print(f"Model loaded successfully with backend: {backend}")
        return True
    except Exception as exc:
        MODEL_ERROR = str(exc)
        print(f"Error loading model: {exc}")
        return False


def stream_reasoning(image, question):
    global model, processor

    if image is None:
        yield None, "Please upload an image."
        return

    if not MODEL_LOADED:
        message = MODEL_ERROR or "Model not loaded. Please restart the app with a valid model path."
        yield image, message
        return

    from threading import Thread
    from transformers import TextIteratorStreamer

    orig_h, orig_w = image.shape[:2]
    overlay_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    inputs, _ = prepare_generation_inputs(processor, image, question, model=model)

    streamer = TextIteratorStreamer(
        processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        skip_prompt=True,
        skip_special_tokens=False,
    )
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=512, use_cache=True)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    generated_text = ""
    ref_pattern = re.compile(r"<ref>([^<]+)</ref><box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>")

    for new_text in streamer:
        generated_text += new_text
        ref_matches = ref_pattern.findall(generated_text)
        current_overlay = overlay_img.copy()

        for match in ref_matches:
            label = match[0]
            x1, y1, x2, y2 = map(int, match[1:])
            abs_x1 = int((x1 / 1000) * orig_w)
            abs_y1 = int((y1 / 1000) * orig_h)
            abs_x2 = int((x2 / 1000) * orig_w)
            abs_y2 = int((y2 / 1000) * orig_h)

            cv2.rectangle(current_overlay, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            label_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            cv2.rectangle(
                current_overlay,
                (abs_x1, max(0, abs_y1 - label_size[1] - 4)),
                (abs_x1 + label_size[0] + 4, abs_y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                current_overlay,
                label,
                (abs_x1 + 2, max(0, abs_y1 - 2)),
                font,
                font_scale,
                (0, 0, 0),
                1,
            )

        final_frame = cv2.cvtColor(current_overlay, cv2.COLOR_BGR2RGB)
        yield final_frame, generated_text

    thread.join()


def create_demo():
    import gradio as gr

    with gr.Blocks(title="V-CoT Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # V-CoT: Grounded Visual Reasoning

        Upload an image and ask a question. The model will explain its reasoning
        step-by-step, highlighting objects it references with bounding boxes.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(label="Input Image", type="numpy")
                inp_txt = gr.Textbox(
                    label="Question",
                    value="Explain step-by-step what you see in this image.",
                    placeholder="Enter your question here...",
                )
                btn = gr.Button("Generate Reasoning", variant="primary")

            with gr.Column(scale=1):
                out_img = gr.Image(label="Live Visualization")
                out_txt = gr.Textbox(label="Reasoning Trace", lines=12)

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


def main():
    parser = argparse.ArgumentParser(description="V-CoT Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/checkpoints",
        help="Path to trained model checkpoint or HuggingFace model ID",
    )
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()

    if not load_model(args.model_path):
        print("\nFailed to load model. Starting demo in limited mode.\n")

    demo = create_demo()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
