import pickle
import json
import numpy as np
import re
from sympy import limit
from tqdm import tqdm
import argparse
import os
from postprocess_util import create_multi_turn_prompt_to_fix_output, parse_claude_output, parse_gpt_output, parse_gemini_output
from best_effort_string_match import find_matched_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--model_output_path', default=None, type=str, required=True, help='the path to read the generation output')
    parser.add_argument('--parsed_output_path', default=None, type=str, required=True, help='the path to store the parsed output')
    parser.add_argument('--original_prompt_dict_path', default=None, type=str, required=True, help='the path to read the original prompt dict')
    parser.add_argument('--regen_prompt_dict_path', default=None, type=str, required=True, help='the path to store regeneration prompt dict')
    parser.add_argument('--limit_k', default=20, type=int, help='the number of sentences the model should output')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()
    model_name = args.model_name
    model_output_path = args.model_output_path
    parsed_output_path = args.parsed_output_path
    original_prompt_dict_path = args.original_prompt_dict_path
    regen_prompt_dict_path = args.regen_prompt_dict_path
    limit_k = args.limit_k
    dataset_path = args.dataset_path

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    hint2opt_k = {}
    for p in dataset:
        hint2opt_k[p['hint']] = p['opt_k']

    with open(model_output_path, "rb") as f:
        model_output = pickle.load(f)
        
    with open(original_prompt_dict_path, 'rb') as f:
        prompt_dict = pickle.load(f)

    if "gpt" in model_name:
        parse_func = parse_gpt_output
    elif "claude" in model_name:
        parse_func = parse_claude_output
    elif "gemini" in model_name:
        parse_func = parse_gemini_output
    else:
        parse_func = find_matched_list
        # raise ValueError("Model name not recognized")
    
    parsed_list_dict = {}
    new_prompt_dict = {}
    for prompt_id in model_output:
        parsed_output = parse_func(prompt_dict[prompt_id], model_output[prompt_id])

        if limit_k == -1:
            curr_limit = hint2opt_k[prompt_id]
        else:
            curr_limit = limit_k
        
        if len(parsed_output) > curr_limit:
            # should generate the new multi-turn prompt for this case

            text_ele_limit = curr_limit

            prompt_template = """You have output more than {text_ele_limit} text elements, which violates the requirement to find no more than {text_ele_limit} number of text elements. You must choose the best {text_ele_limit} text elements in the same format as the previous output."""

            curr_prompt = create_multi_turn_prompt_to_fix_output(prompt_id, prompt_template, original_prompt_dict_path, model_output_path)
            new_prompt_dict[prompt_id] = curr_prompt

       
        parsed_list_dict[prompt_id] = parsed_output

    # assert len(new_prompt_dict) + len(parsed_list_dict) == len(model_output)
    print(f"{len(new_prompt_dict)} prompts need to be regenerated")

    with open(regen_prompt_dict_path, "wb") as f:
        pickle.dump(new_prompt_dict, f)

    with open(parsed_output_path, "wb") as f:
        pickle.dump(parsed_list_dict, f)
