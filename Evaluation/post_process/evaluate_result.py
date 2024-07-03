import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm
import warnings
from postprocess_util import get_west_coast_time, append_row_to_csv
import pdb
import json

if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--input_path', default=None, type=str, required=True, help='the path to the parsed generation output')
    parser.add_argument('--dataset_path', default=None, type=str, required=True, help='the path to the dataset')
    parser.add_argument('--exp_name', type=str, required=True, help='the experiment name')
    parser.add_argument('--logging_csv', type=str, required=True, help='the path to the logging csv file')
    parser.add_argument('--limit_k', default=20, type=int, help='the number of sentences the model should output')
    # parser.add_argument('--output_path', default=None, type=str, required=True, help='the path to the parsed output')

    args = parser.parse_args()
    model_name = args.model_name
    input_path = args.input_path
    dataset_path = args.dataset_path
    exp_name = args.exp_name
    logging_csv_path = args.logging_csv
    sent_limit = args.limit_k

    with open(input_path, "rb") as f:
        parsed_list_dict = pickle.load(f)
    
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    cov_list = []

    for k,p in tqdm(dataset.items()):
        # TODO: Make sure to change this after the dataset is done !!!!!!!!!!!!!!!!!!!
        
        key = k
        model_picked_sents_ind = parsed_list_dict[k]
        if len(model_picked_sents_ind) == 0:
            print('!'*20)
            print("Warning: the model is retrieving 0 sentences. CHECK the model output!!!")
            print(f"Hint: {key}")
            print('!'*20)

        if sent_limit == -1:
            # print('Using optimal_k')
            curr_sent_limit = p['evidence_retrieval_at_optimal_evaluation']['optimal']

            if len(model_picked_sents_ind) <= curr_sent_limit:
                maximum_covered_asp = []
                model_covered_asp = []

                for s in p['sentence_index2aspects']:
                    if s not in p['results_aspect_list_ids']:
                        continue
                    maximum_covered_asp += p['sentence_index2aspects'][s]
                    if int(s) in model_picked_sents_ind:
                        model_covered_asp += p['sentence_index2aspects'][s]
                
                maximum_covered_asp = set(maximum_covered_asp)
                model_covered_asp = set(model_covered_asp)

                if len(maximum_covered_asp) == 0:
                    warnings.warn(f"Should NOT happen: Maximum covered aspect is 0 for {key}")
                    continue

                cov_list.append(len(model_covered_asp)/len(maximum_covered_asp))

            else:
                maximum_covered_asp = []

                for s in p['sentence_index2aspects']:
                    maximum_covered_asp += p['sentence_index2aspects'][s]
                maximum_covered_asp = set(maximum_covered_asp)

                
                if len(maximum_covered_asp) == 0:
                    warnings.warn(f"Should NOT happen: Maximum covered aspect is 0 for {key}")
                    continue
                temp_list = []
                for _ in range(100000):
                    model_covered_asp = []
                    random.shuffle(model_picked_sents_ind)

                    for s in model_picked_sents_ind[:curr_sent_limit]:
                        model_covered_asp += p['sentence_index2aspects'][s]

                    model_covered_asp = set(model_covered_asp)

                    temp_list.append(len(model_covered_asp)/len(maximum_covered_asp))

                cov_list.append(float(np.mean(temp_list)))
        else:
            # this is the case for fixed sent_limit
            # maximum_covered_asp_num = len(p['best_solutions'][sent_limit]['best_covered_aspects'])

            maximum_covered_asp = []

            for s in p['sentence_index2aspects']:
                maximum_covered_asp += p['sentence_index2aspects'][s]
            maximum_covered_asp = set(maximum_covered_asp)

            
            if len(maximum_covered_asp) == 0:
                warnings.warn(f"Should NOT happen: Maximum covered aspect is 0 for {key}")
                continue

            maximum_covered_asp_num = len(maximum_covered_asp)

            if len(model_picked_sents_ind) <= sent_limit:
                # maximum_covered_asp = []
                model_covered_asp = []

                for s in p['sentence_index2aspects']:
                    # maximum_covered_asp += p['sentence_index2aspects'][s]
                    if s in model_picked_sents_ind:
                        model_covered_asp += p['sentence_index2aspects'][s]
                
                # maximum_covered_asp = set(maximum_covered_asp)
                
                model_covered_asp = set(model_covered_asp)

                if maximum_covered_asp_num == 0:
                    warnings.warn(f"Should NOT happen: Maximum covered aspect is 0 for {key}")
                    continue

                cov_list.append(len(model_covered_asp)/maximum_covered_asp_num)

            else:
                # maximum_covered_asp = []

                # for s in p['sentence_index2aspects']:
                #     maximum_covered_asp += p['sentence_index2aspects'][s]
                # maximum_covered_asp = set(maximum_covered_asp)
                if maximum_covered_asp_num == 0:
                    warnings.warn(f"Should NOT happen: Maximum covered aspect is 0 for {key}")
                    continue
                temp_list = []
                for _ in range(100000):
                    model_covered_asp = []
                    random.shuffle(model_picked_sents_ind)

                    for s in model_picked_sents_ind[:sent_limit]:
                        model_covered_asp += p['sentence_index2aspects'][s]

                    model_covered_asp = set(model_covered_asp)

                    temp_list.append(len(model_covered_asp)/maximum_covered_asp_num)

                cov_list.append(float(np.mean(temp_list)))

    
    # print(cov_list)
    print(f"Model: {model_name}")
    print(f"Input: {input_path}")
    print(f"Average coverage: {np.mean(cov_list)}")

    result_dict = {"Experiment Name": exp_name ,"Model": model_name, "Cov@": sent_limit, "Average Coverage": np.mean(cov_list),  "Input": input_path, "Coverage List": cov_list, "Time": get_west_coast_time()}

    append_row_to_csv(logging_csv_path, result_dict)

    



