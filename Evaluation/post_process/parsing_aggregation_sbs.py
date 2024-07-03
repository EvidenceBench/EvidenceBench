import pickle
import json
import numpy as np
import re
from sympy import limit
from tqdm import tqdm
import argparse
import os
from postprocess_util import create_multi_turn_prompt_to_fix_output, parse_claude_output, parse_gpt_output, parse_gemini_output, parse_llama_output, parse_hyena_output, parse_gemini_output_SBS
from best_effort_string_match import find_matched_list

def aggregate_results(pid2cid, parsing_output):
    collected_results = {}
    for k in parsing_output:
        parsed_list = parsing_output[k]
        if k[0] not in collected_results:
            collected_results[k[0]] = []
        mapping = pid2cid[k]
        
        mapped_idx = [mapping[sub_k] for sub_k in parsed_list if sub_k in pid2cid[k]]

        for sub_k in parsed_list:
            if sub_k not in pid2cid[k]:
                mapped_idx = []

        collected_results[k[0]] += mapped_idx
        collected_results[k[0]] = list(set(collected_results[k[0]]))
        
    return collected_results

def create_sbs_second_prompt(dataset, aggregation_result, prompt_template, text_ele_limit):
    prompt_dict = {}
    all_idx_mapping = {}
    hint2opt_k = {}
    for key,p in dataset.items():
        hint2opt_k[key] = p['evidence_retrieval_at_optimal_evaluation']['optimal']
    
    for k in aggregation_result:
        for key,p in dataset.items():
            if key == k:
                point = p
                break
        if text_ele_limit == -1:
            curr_limit = hint2opt_k[k]
        else:
            curr_limit = text_ele_limit
        idx_list = sorted(aggregation_result[k])
        prompt_idx_text = ""
        count = 1
        new_idx_mapping = {}
        for sub_k in idx_list:
            prompt_idx_text += f"{count}: {point['paper_as_candidate_pool'][int(sub_k)]}" + "\n\n"
            new_idx_mapping[count] = sub_k
            count += 1
        prompt_dict[k] = prompt_template.format(hypothesis=point['hypothesis'], cand_pool=prompt_idx_text, text_ele_limit=curr_limit)
        all_idx_mapping[k] = new_idx_mapping
    return prompt_dict, all_idx_mapping

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--model_output_path', default=None, type=str, required=True, help='the path to read the generation output')
    parser.add_argument('--parsed_mapping_path', default=None, type=str, required=True, help='the path to store the parsed output')
    parser.add_argument('--original_prompt_dict_path', default=None, type=str, required=True, help='the path to read the original prompt dict')
    parser.add_argument('--regen_prompt_dict_path', default=None, type=str, required=True, help='the path to store regeneration prompt dict')
    parser.add_argument('--dataset_path', default=None, type=str, required=True, help='the path to dataset')
    parser.add_argument('--prompt_template_name', default=None, type=str, required=True, help='the prompt template to use')
    parser.add_argument('--limit_k', default=20, type=int, help='the number of sentences the model should output')


    args = parser.parse_args()
    model_name = args.model_name
    model_output_path = args.model_output_path
    parsed_mapping_path = args.parsed_mapping_path
    original_prompt_dict_path = args.original_prompt_dict_path
    regen_prompt_dict_path = args.regen_prompt_dict_path
    dataset_path = args.dataset_path
    prompt_template_name = args.prompt_template_name
    limit_k = args.limit_k
    
    with open(model_output_path, "rb") as f:
        model_output = pickle.load(f)
        
    with open(f"{original_prompt_dict_path}/prompt.pickle", 'rb') as f:
        prompt_dict = pickle.load(f)
        
    with open(f"{original_prompt_dict_path}/pid2cid.pickle", 'rb') as f:
        pid2cid = pickle.load(f)
        
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    
    if "gpt" in model_name:
        parse_func = parse_gpt_output
    elif "claude" in model_name:
        parse_func = parse_claude_output
    elif "gemini" in model_name:
        parse_func = parse_gemini_output_SBS
    elif "llama" in model_name:
        parse_func = parse_llama_output
    elif "Hyena" in model_name:
        parse_func = parse_hyena_output
    else:
        parse_func = find_matched_list
        # raise ValueError("Model name not recognized")
    
    parsed_list_dict = {}
    for prompt_id in tqdm(model_output):
        parsed_output = parse_func(prompt_dict[prompt_id], model_output[prompt_id])
        parsed_list_dict[prompt_id] = parsed_output
        
    aggregation_result = aggregate_results(pid2cid, parsed_list_dict)
        
    with open("./pre_process/prompt_template.json", 'r') as f:
        prompt_template = json.load(f)
        
    new_prompt_dict, all_idx_mapping = create_sbs_second_prompt(dataset, aggregation_result, prompt_template[prompt_template_name], limit_k)

    print(f"-Writing new file {regen_prompt_dict_path}")
    print(f"-Writing new file {parsed_mapping_path}")
    
    with open(regen_prompt_dict_path, "wb") as f:
        pickle.dump(new_prompt_dict, f)
        
    with open(parsed_mapping_path, "wb") as f:
        pickle.dump(all_idx_mapping, f)
        
    
        
        