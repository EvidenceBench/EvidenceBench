import pickle
import json
import numpy as np
import argparse
from scipy.spatial.distance import cosine

def calculate_similarity(query_emb, doc_emb):
    # Note: cosine from scipy is calculated by 1 - (dot product / norm)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine 
    return 1 - cosine(query_emb, doc_emb)

def get_top_k_docs(dataset, emb_dict, limit_k):
    all_top_k_idx = {}
    for k,point in dataset.items():
        key = k
        query = point['hypothesis']
        query_emb = emb_dict[query]
        doc_similarity = {}
        for idx, k in enumerate(point['paper_as_candidate_pool']):
            doc_similarity[idx] = calculate_similarity(query_emb, emb_dict[k])
        sorted_sim = dict(sorted(doc_similarity.items(), key=lambda x: x[1], reverse=True))
        if limit_k == -1:
            # this is the case for optimal k
            curr_k = point['evidence_retrieval_at_optimal_evaluation']['optimal']
            top_k_idx = list(sorted_sim.keys())[:curr_k]
        else:
            # this is the case for fixed k
            top_k_idx = list(sorted_sim.keys())[:limit_k]
        all_top_k_idx[key] = top_k_idx
    return all_top_k_idx

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--model_output_path', default=None, type=str, required=True, help='the path to read the generation output')
    parser.add_argument('--parsed_output_path', default=None, type=str, required=True, help='the path to store the parsed output')
    parser.add_argument('--dataset_path', default=None, type=str, required=True, help='the path to dataset')
    parser.add_argument('--limit_k', default=20, type=int, help='the number of sentences the model should output')
    # parser.add_argument('--instruction_name', default=None, type=str, required=True, help="the instruction used")


    args = parser.parse_args()
    model_name = args.model_name
    model_output_path = args.model_output_path
    parsed_output_path = args.parsed_output_path
    dataset_path = args.dataset_path
    limit_k = args.limit_k
    # instruction_name = args.instruction_name
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(model_output_path, 'rb') as f:
        embedding_output = pickle.load(f)
        
    # instruction = instruction_temp[instruction_name]
    all_top_k_idx = get_top_k_docs(dataset, embedding_output, limit_k)
    
    print("Finished scoring...")
    print(f"-Writing results to {parsed_output_path}")
    with open(parsed_output_path, 'wb') as f:
        pickle.dump(all_top_k_idx, f)
