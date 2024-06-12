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
    parser.add_argument('--output_path', default="", type=str, help='Path to the dir of the output file')
    parser.add_argument('--prompt_dict_name', type=str, help='Name of the prompt_dict', required=True)
    parser.add_argument('--max_cand_pool_token', type=int, help="Maximum number of tokens in the candidate pool", default=-1) # if -1 then there is no limit
    parser.add_argument("--with_images", type=str, help='Whether the prompt_dict contains paths of images', default="False")
    parser.add_argument("--special_requirements", type=str, help='Special requirements for the prompt_dict', default="")
    parser.add_argument("--limit", type=int, help='the max number of sentences to retrieve, -1 means optimal k', default=-1)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    dataset_path = args.dataset
    prompt_template_name = args.prompt_template_name
    output_path = args.output_path
    prompt_dict_name = args.prompt_dict_name
    max_cand_pool_token = args.max_cand_pool_token
    with_image = (args.with_images == "True")
    special_requirements = args.special_requirements
    limit = args.limit

    if special_requirements not in ['', 'no_hypothesis', 'icl1', 'icl2', 'cot', 'abstract_only', 'no_abstract']:
        print(f"special_requirements must be one of ['', 'no_hypothesis', 'icl1', 'icl2', 'cot', 'abstract_only', 'no_abstract'], the current value is {special_requirements}")

        exit(1)

    

    dataset_name = os.path.splitext(dataset_path)[0]
    if not dataset_path.endswith('.pickle'):
        print(f"The dataset is not a pickle")
    print(f"Dataset name: {dataset_name}")

    # Load the dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    if special_requirements == 'no_hypothesis':
        print("!!! No hypothesis, please make sure the prompt template is correct !!!")
        for point in dataset:
            point['hypothesis'] = ""

    elif special_requirements == 'icl1':

        print("TODO: implement ICL1")

    elif special_requirements == 'icl2':

        print("TODO: implement ICL2")

    elif special_requirements == 'cot':

        print("TODO: implement COT")

    elif special_requirements == 'abstract_only':

        print("!!! No hypothesis, please make sure the prompt template is correct !!!")
        for point in dataset:
            point['hypothesis'] = ""
    

    if max_cand_pool_token != -1:
        if not os.path.exists(f"{dataset_name}_{max_cand_pool_token}.pickle"):

            print(f"Shortening the candidate pool to {max_cand_pool_token} tokens")

            print(f"Before shortening, the number of points in the dataset: {len(dataset)}")

            short_cand_pool = shorten_candidate_pool_for_dataset(dataset, max_cand_pool_token)
            new_dataset = []

            for ind, point in enumerate(dataset):
                new_point = {}
                
                curr_short_cand_pool = short_cand_pool[ind]
                if curr_short_cand_pool is None:
                    continue
                new_cand_pool = []
                new_sent2asp = {}
                new_asp2sent = {}
                old_cand_id2new_cand_id = {}
                
                for ind, cand_pool_id in enumerate(curr_short_cand_pool):
                    new_cand_pool.append(point['candidate_pool'][cand_pool_id])
                    new_sent2asp[ind] = point['sent2aspect'][cand_pool_id]
                    old_cand_id2new_cand_id[cand_pool_id] = ind

                for asp_id in point['aspect2sent']:
                    new_asp2sent[asp_id] = []
                    for cand_id in point['aspect2sent'][asp_id]:
                        if cand_id in curr_short_cand_pool:
                            new_asp2sent[asp_id].append(old_cand_id2new_cand_id[cand_pool_id])

                for k in point:
                    new_point[k] = point[k]
                
                new_point['candidate_pool'] = new_cand_pool
                new_point['aspect2sent'] = new_asp2sent
                new_point['sent2aspect'] = new_sent2asp
                new_point['old_cand_id2new_cand_id'] = old_cand_id2new_cand_id

                new_dataset.append(new_point)

            with open(f"{dataset_name}_{max_cand_pool_token}.pickle", 'wb') as f:
                pickle.dump(new_dataset, f)

            dataset = new_dataset
            print(f"After shortening, the number of points in the dataset: {len(dataset)}")
            print(f"The new dataset is stored at {dataset_name}_{max_cand_pool_token}.pickle")
        else:
            print(f"Loading the shortened dataset from {dataset_name}_{max_cand_pool_token}.pickle")
            with open(f"{dataset_name}_{max_cand_pool_token}.pickle", 'rb') as f:
                dataset = pickle.load(f)
    

    with open("prompt_template_path", 'r') as f:
        prompt_template = json.load(f)

    try:
        curr_prompt_template = prompt_template[prompt_template_name]
    except KeyError:
        print(f"The prompt_template.json file does not have a {prompt_template_name} key")
        exit(1)

    prompt_dict = {}
    if not with_image:
        for ind, p in enumerate(dataset):

            assert p['hypothesis'] == p['central_hypothesis']
            cand_pool_str = ""
            curr_count = 0

            for i in p['candidate_pool']:
                cand_pool_str += f"\n{curr_count}: {i}\n"
                curr_count += 1
            if limit == -1:
                # this is the case for optimal_k
                curr_limit = p['opt_k']
            else:
                curr_limit = limit

            prompt_dict[p['hint']] = curr_prompt_template.format(text_ele_limit=curr_limit, hypothesis=p['hypothesis'], cand_pool=cand_pool_str)
    else:
        print("!!! With Images !!!")
        for ind, p in enumerate(dataset):
            cand_pool_str = ""
            curr_count = 0

            for i in p['candidate_pool']:
                cand_pool_str += f"\n{curr_count}: {i}\n"
                curr_count += 1

            curr_img_path_list = find_img_path_for_point(p)

            prompt_dict[p['hint']] = {}

            if limit == -1:
                # this is the case for optimal_k
                curr_limit = p['opt_k']
            else:
                curr_limit = limit

            prompt_dict[p['hint']]['prompt_text'] = curr_prompt_template.format(text_ele_limit=curr_limit, hypothesis=p['hypothesis'], cand_pool=cand_pool_str)
            prompt_dict[p['hint']]['images'] = curr_img_path_list


    with open(f"{output_path}/{prompt_dict_name}.pickle", "wb") as f:
        pickle.dump(prompt_dict, f)
