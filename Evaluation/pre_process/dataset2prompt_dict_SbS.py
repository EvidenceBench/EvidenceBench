import pickle
import argparse
import json
from preprocess_util import create_prompt_section_by_section
import os


# This program will take a dataset, an evaluation model, and generate a prompt_dict for the dataset. The generated prompt_dict will be stored to the directory provided with name "{dataset_name}_{model_name}.pickle"

import os

def create_directory_if_not_exists(dir_path):
    """
    Creates a directory at the specified path if it does not exist.
    
    Parameters:
    dir_path (str): The path to the directory to be created.
    """
    # Check if the directory already exists
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory created at: {dir_path}")
    else:
        raise ValueError(f"Directory already exists at: {dir_path}")
        # print("already exist")

if __name__ == "__main__":
    # args = params()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--prompt_template_name', type=str, help='the prompt template to load', required=True)
    parser.add_argument('--output_path', required=True, type=str, help='Path to the dir of the output file')
    parser.add_argument('--prompt_dict_name', type=str, help='Name of the prompt_dict', required=True)
    parser.add_argument('--limit_k', default=10, type=int, help='the number of sentences the model should output')
    # parser.add_argument('--max_cand_pool_token', type=int, help="Maximum number of tokens in the candidate pool", default=-1) # if -1 then there is no limit
    # parser.add_argument("--with_images", type=bool, help='Whether the prompt_dict contains paths of images', default=False)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    dataset_path = args.dataset
    prompt_template_name = args.prompt_template_name
    output_path = args.output_path
    prompt_dict_name = args.prompt_dict_name
    limit_k = args.limit_k
    # max_cand_pool_token = args.max_cand_pool_token
    # with_image = args.with_images

    dataset_name = os.path.splitext(dataset_path)[0]
    if not dataset_path.endswith('.pickle'):
        print(f"The dataset is not a pickle")
    print(f"Dataset name: {dataset_name}")

    # Load the dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open("/data/Elsevier/Evaluation/pre_process/prompt_template.json", 'r') as f:
        prompt_template = json.load(f)

    sbs_prompt_template = prompt_template[prompt_template_name]

    # TODO: Make sure text_ele_limit is added to the pipeline

    prompt_dict, prompt_id2cand_id = create_prompt_section_by_section(dataset, sbs_prompt_template, limit_k)

    create_directory_if_not_exists(f"{output_path}/{prompt_dict_name}")




    with open(f"{output_path}/{prompt_dict_name}/prompt.pickle", "wb") as f:
        pickle.dump(prompt_dict, f)
    with open(f"{output_path}/{prompt_dict_name}/pid2cid.pickle", "wb") as f:
        pickle.dump(prompt_id2cand_id, f)

    