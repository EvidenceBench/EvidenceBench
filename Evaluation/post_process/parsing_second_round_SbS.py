import pickle
import json
import numpy as np
import re
from sympy import limit
from tqdm import tqdm
import argparse
import os
from postprocess_util import create_multi_turn_prompt_to_fix_output, parse_claude_output, parse_gpt_output, parse_gemini_output, parse_llama_output, parse_hyena_output
from best_effort_string_match import find_matched_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--model_output_path', default=None, type=str, required=True, help='the path to read the generation output')
    parser.add_argument('--parsed_output_path', default=None, type=str, required=True, help='the path to store the parsed output')
    parser.add_argument('--pid2cid_path', type=str, required=True, help='the path to store pid2cid dict')
    parser.add_argument('--original_prompt_dict_path', default=None, type=str, required=True, help='the path to read the original prompt dict')

    args = parser.parse_args()
    model_name = args.model_name
    model_output_path = args.model_output_path
    parsed_output_path = args.parsed_output_path
    original_prompt_dict_path = args.original_prompt_dict_path
    pid2cid_path = args.pid2cid_path

    with open(pid2cid_path, 'rb') as f:
        pid2cid = pickle.load(f)
    
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
    elif "llama" in model_name:
        parse_func = parse_llama_output
    elif "Hyena" in model_name:
        parse_func = parse_hyena_output
    else:
        parse_func = parse_llama_output
        # raise ValueError("Model name not recognized")
    
    parsed_list_dict = {}
    for prompt_id in model_output:
        parsed_output = parse_func(prompt_dict[prompt_id], model_output[prompt_id])
        mapped_parsed_output = []
        for pid in parsed_output:
            if pid not in  pid2cid[prompt_id]:
                continue
            mapped_parsed_output.append(int(pid2cid[prompt_id][pid]))
        parsed_list_dict[prompt_id] = mapped_parsed_output

    if os.path.exists(parsed_output_path):
        print("The parsed output file already exists. Overwriting some of the results")
        
        with open(parsed_output_path, 'rb') as f:
            old_parsed_list_dict = pickle.load(f)
        
    else:
        raise RuntimeError("This case should not happen. The parsed output file must have been created in the first round.")
    
    for k in parsed_list_dict:
        old_parsed_list_dict[k] = parsed_list_dict[k]

    print(f"Overwritten {len(parsed_list_dict)} entries in the parsed output file.")


    with open(parsed_output_path, "wb") as f:
        pickle.dump(old_parsed_list_dict, f)

