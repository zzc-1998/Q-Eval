import argparse
import torch
import time
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from collections import defaultdict
import numpy as np

import os

def wa5(logits):
    import numpy as np
    logprobs = np.array([logits["Excellent"], logits["Good"], logits["Fair"], logits["Poor"], logits["Bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def main(args):
    # Model
    disable_torch_init()
    device = args.device
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_base, torch_dtype=torch.float16, device_map=args.device
    )
    if args.model_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print('Convert to FP16...')
        model.to(torch.float16)
    
    processor = Qwen2VLProcessor.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/OPEN_SOURCE_MODEL/Qwen2-VL-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    
    prompt_image = "A zigzag path leading to a circular fountain."
    image = "test.png"

    prompt_video = "fantasy background and the camera zooms out"
    video = "0ad3d439_88.mp4"
    
    # image assessment
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    # 图像质量
                    # "text": f"<image>Suppose you are an expert in evaluating the visual quality of AI-generated image. First identify any visual distortions and positive visual appeal regarding low-level features and aesthetics. Next, assess the severity of distortions and their impact on the viewing experience, noting whether they are subtle or distracting, and evaluate how the positive features enhance the image's visual appeal, considering their strength and contribution to the overall aesthetics. Finally, balance the identified distortions against the positive aspects and give your rating on the visual quality. \nYour rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad.\nFor this image, the text prompt is \"{prompt}\". Now please rate this image:"
                    # 图文一致性
                    "text": f"<image>Suppose you are an expert in evaluating the alignment between the text prompt and the AI-generated image. Begin by considering whether the overall concept of the prompt is captured in the image. Then, examine the specific details, such as the presence of key objects, their attributes, and relationships. Check if the visual content accurately reflects these aspects. Finally, give your alignment rating considering both overall and detailed accuracy.\nYour rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad.\nFor this image, the text prompt is \"{prompt_image}\". Now please rate this image:"
                },
            ],
        }
    ]
    

    '''
    # video assessment
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
                    "max_pixels": 100352,
                    "fps": 8,
                },
                {
                    "type": "text", 
                    # 一致性
                    "text": f"<video>Suppose you are an expert in evaluating the alignment between the text prompt and the AI-generated video. Begin by considering whether the overall concept of the prompt is captured in the video. Then, examine the specific details, such as the presence of key objects, their attributes, and relationships. Check if the visual content accurately reflects these aspects. Finally, give your alignment rating considering both overall and detailed accuracy.\nYour rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad.\nFor this video, the text prompt is \"{prompt_video}\". Now please rate this video:"
                    # 质量
                    # "text": f"<video>Suppose you are an expert in evaluating the visual quality of AI-generated video. First identify any visual distortions and positive visual appeal regarding low-level features and aesthetics. Next, assess the severity of distortions and their impact on the viewing experience, noting whether they are subtle or distracting, and evaluate how the positive features enhance the video's visual appeal, considering their strength and contribution to the overall aesthetics. Finally, balance the identified distortions against the positive aspects and give your rating on the visual quality. \nYour rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad.\nFor this video, the text prompt is \"{prompt_name}\". Now please rate this video:"
                },
            ],
        }
    ]
    '''
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    toks = ["Good", "Poor", "Fair", "Excellent", "Bad"]
    ids_ = [id_[0] for id_ in tokenizer(toks)["input_ids"]]


    with torch.inference_mode():
        output_logits = model(**inputs)["logits"][:,-1]

    logits = defaultdict(float)
    for tok, id_ in zip(toks, ids_):
        logits[tok] += output_logits[0, id_].item()

    predict_score = wa5(logits)
    print(predict_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/koutengchuan/ms-swift-main/Q_Eval_Score_Models/output_video_alignment/qwen2-vl-7b-instruct/v0-20250529-114003/checkpoint-6282-merged")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
