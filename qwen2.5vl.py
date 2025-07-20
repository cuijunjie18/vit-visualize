import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import argparse 
import time

img_dir = "demo_images/catdog.png"

def model_inference():
    start = time.perf_counter()
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "pretrained/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
        device_map="auto",
    )

    # breakpoint()

    # default processor
    processor = AutoProcessor.from_pretrained("pretrained/Qwen2.5-VL-3B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{img_dir}",
                },
                {"type": "text", "text": "Find the dog in figure"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking = True, # 设置思考
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    end = time.perf_counter()
    print(f"执行时间: {end - start:.8f} 秒")  # 小数点后8位精度
    return output_text

if __name__ == "__main__":
    output_text = model_inference()
    text = output_text[0]
    print(text)
