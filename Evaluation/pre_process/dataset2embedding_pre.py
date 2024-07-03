import pickle
import argparse
import json


if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--instruction_name', type=str, help='the instruction added before the hypothesis', required=True)
    parser.add_argument('--output_path', default="/data/Elsevier/embedding/text_to_emb/", type=str, help='Path to the dir of the output formatted file')
    parser.add_argument('--exp_name', type=str, help='Name of the experiment', required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    dataset_path = args.dataset
    instruction_name = args.instruction_name
    output_path = args.output_path
    exp_name = args.exp_name

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open("pre_process/instruction_for_emb.json", 'r') as f:
        instructions = json.load(f)

    curr_instruction = instructions[instruction_name]

    query_instruction = curr_instruction['query']
    candidate_instruction = curr_instruction['candidate']

    text_to_emb = []

    query_dict = {}
    doc_dict = {}

    for point in dataset:
        
        query_dict[point['hypothesis']] = (query_instruction.format(hypothesis=point['hypothesis']))
        for candidate in point['paper_as_candidate_pool']:
            doc_dict[candidate] = candidate_instruction.format(candidate=candidate)

    with open(f"/data/Elsevier/Evaluation/embedding/text_to_emb/{exp_name}.pickle", "wb") as f:
        pickle.dump({'query': query_dict, 'candidate': doc_dict}, f)

    


    

