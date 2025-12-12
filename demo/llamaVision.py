'''
CUDA_VISIBLE_DEVICES=1 python ./demo/llamaVision.py
'''

# import requests
from dataclasses import dataclass
import PIL
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

@dataclass
class VisPrompt:
    images: PIL.Image.Image | list[PIL.Image.Image]
    user_text: str
    assistant_text: None | str = None
"""
Run with:
CUDA_VISIBLE_DEVICES=1 python ./demo/llamaVision.py
"""

USE_FEWSHOT = True  # set to False to test zero-shot


MESSAGES_FS = [
        {
            "role": "system",
            "content": (
                "You are a helpful vision assistant. "
                "Answer in plain English with a short sentence that names the animal. "
                "If you are not sure, say 'I am not sure.'"
            ),
        },

        # EXAMPLE 1: cat
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In plain English, what animal is in this image?"},
            ],
        },
        {"role": "assistant", "content": "This is a cat."},

        # EXAMPLE 2: dog
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In plain English, what animal is in this image?"},
            ],
        },
        {"role": "assistant", "content": "This is a dog."},

        # EXAMPLE 3: lion
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In plain English, what animal is in this image?"},
            ],
        },
        {"role": "assistant", "content": "This is a lion."},

        # QUERY: cat (no assistant answer yet!)
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In plain English, what animal is in this image?"},
            ],
        },
    ]

MESSAGES_ZS = [
        {
            "role": "system",
            "content": (
                "You are a helpful vision assistant. "
                "Answer in plain English with a short sentence that names the animal. "
                "If you are not sure, say 'I am not sure.'"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In plain English, what animal is in this image?"},
            ],
        },
    ]

MESSAGES = MESSAGES_FS if USE_FEWSHOT else MESSAGES_ZS

def main():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,  # your transformers warns that torch_dtype is deprecated
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------
    ex1_path = "./demo/images/ex1.jpg"   # axolotl
    ex2_path = "./demo/images/ex2.jpg"   # LEGO lion
    ex3_path = "./demo/images/ex3.jpg"   # dog
    query_path = "./demo/images/bear.jpg"  # grey cat

    ex1_image = Image.open(ex1_path).convert("RGB")
    ex2_image = Image.open(ex2_path).convert("RGB")
    ex3_image = Image.open(ex3_path).convert("RGB")
    query_image = Image.open(query_path).convert("RGB")

    if USE_FEWSHOT:
        images = [ex1_image, ex2_image, ex3_image, query_image]
    else:
        images = [query_image]

    # --------------------------------------------------------
    # Build prompt and inputs
    # --------------------------------------------------------
    input_text = processor.apply_chat_template(
        MESSAGES,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=images,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    # --------------------------------------------------------
    # Generate (greedy, deterministic) and decode continuation
    # --------------------------------------------------------
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        # do_sample=False,
        temperature=0.1,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_len:]

    raw_answer = processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Strip leading "assistant" header if present
    if raw_answer.lower().startswith("assistant"):
        # Remove first line that contains 'assistant'
        raw_answer = "\n".join(raw_answer.splitlines()[1:]).strip()

    print("MODE:", "few-shot" if USE_FEWSHOT else "zero-shot")
    print("MODEL ANSWER:", raw_answer)


if __name__ == "__main__":
    main()
