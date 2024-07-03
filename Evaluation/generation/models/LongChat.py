import torch
from fastchat.model import load_model, get_conversation_template, add_model_args
import pickle
import os
import sys
from .util import check_and_create_directory
from tqdm import tqdm

def generate_longchat_response(prompt_dict, temperature, max_tokens, output_path):
    model_path = "lmsys/longchat-7b-32k-v1.5"
    device = "cuda"
    num_gpus = 1

    # prompts = [prompt_dict[key] for key in prompt_dict]

    model, tokenizer = load_model(
    model_path,
    device=device,
    num_gpus=num_gpus,
    # max_gpu_memory=args.max_gpu_memory,
    # load_8bit=args.load_8bit,
    # cpu_offloading=args.cpu_offloading,
    # revision=args.revision,
    # debug=args.debug,
    )
    output_dict = {}

    for key in tqdm(prompt_dict):
        msg = prompt_dict[key]
        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=1,
            max_new_tokens=max_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        
        
        output_dict[key] = outputs

    check_and_create_directory(f"./{output_path}")

    with open(f"./{output_path}/collected_results.pickle","wb") as f:
        pickle.dump(output_dict,f)