import pickle
import argparse
import json
from preprocess_util import shorten_candidate_pool_for_dataset, find_img_path_for_point
import os


# This program will take a dataset, an evaluation model, and generate a prompt_dict for the dataset. The generated prompt_dict will be stored to the directory provided with name "{dataset_name}_{model_name}.pickle"

if __name__ == "__main__":
    # args = params()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--prompt_template_name', type=str, help='the prompt template to load', required=True)
    parser.add_argument('--output_path', default="/data/Elsevier/Evaluation/generation/prompts/", type=str, help='Path to the dir of the output file')
    parser.add_argument('--prompt_dict_name', type=str, help='Name of the prompt_dict', required=True)
    parser.add_argument("--limit", type=int, help='the max number of sentences to retrieve, -1 means optimal k', default=-1)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    dataset_path = args.dataset
    prompt_template_name = args.prompt_template_name
    output_path = args.output_path
    prompt_dict_name = args.prompt_dict_name
    limit = args.limit

    

    dataset_name = os.path.splitext(dataset_path)[0]
    if not dataset_path.endswith('.json'):
        print(f"The dataset is not a json")
    print(f"Dataset name: {dataset_name}")

    # Load the dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
      

    with open("/data/Elsevier/Evaluation/pre_process/prompt_template.json", 'r') as f:
        prompt_template = json.load(f)

    try:
        curr_prompt_template = prompt_template[prompt_template_name]
    except KeyError:
        print(f"The prompt_template.json file does not have a {prompt_template_name} key")
        exit(1)

    prompt_dict = {}
    for k, p in dataset.items():

        cand_pool_str = ""
        curr_count = 0

        for i in p['paper_as_candidate_pool']:
            cand_pool_str += f"\n{curr_count}: {i}\n"
            curr_count += 1
        if limit == -1:
            # this is the case for optimal_k
            curr_limit = p['results_evidence_retrieval_at_optimal_evaluation']['optimal']
        else:
            curr_limit = limit

        prompt_dict[k] = curr_prompt_template.format(text_ele_limit=curr_limit, hypothesis=p['hypothesis'], cand_pool=cand_pool_str)

    with open(f"{output_path}/{prompt_dict_name}.pickle", "wb") as f:
        pickle.dump(prompt_dict, f)

    