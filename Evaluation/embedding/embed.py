import pickle
import argparse
from embedding_models.local_emb import get_embedding_dict_local
from embedding_models.openai_based import emb_text_in_parallel_openai
from embedding_models.voyage_ai_related import embed_texts_with_voyage
import pdb

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--text_path', type=str, help='Path to the text to emb')
    parser.add_argument('--model_name', type=str, help='the name of the model used for embedding', required=True)
    parser.add_argument('--output_path', default="/data/Elsevier/Evaluation/embedding/embedding_results/", type=str, help='Path to the dir of the output file')
    parser.add_argument('--exp_name', type=str, help='Name of the experiment', required=True)
    parser.add_argument('--use_api', type=str, help='Whether to use the API to embed the text', default=False)
    parser.add_argument('--cuda', type=str, help='The string that describes the cuda to use for embedding', default="0")
    parser.add_argument('--batch_size', type=int, help='batch size for the embedding', default=256)


    # Parse the command line arguments
    args = parser.parse_args()

    text_path = args.text_path
    model_name = args.model_name
    output_path = args.output_path
    exp_name = args.exp_name
    use_api = (args.use_api == 'True')
    cuda = args.cuda
    batch_size = args.batch_size

    with open(text_path, "rb") as f:
        text_to_emb = pickle.load(f)

    query_to_emb = text_to_emb['query']
    doc_to_emb = text_to_emb['candidate']

    # text_to_emb is a dict key as the original text (hypothesis/candidates) from the dataset, value as the text to be embedded

    emb2query = {}

    for k, v in query_to_emb.items():
        emb2query[v] = k

    emb2doc = {}
    for k, v in doc_to_emb.items():
        emb2doc[v] = k

    if use_api:
        if 'voyage' in model_name:
            embed_func = embed_texts_with_voyage
        else:
            embed_func = emb_text_in_parallel_openai
    else:
        embed_func = get_embedding_dict_local

    
    query_embedding_dict = embed_func(all_text_list=list(query_to_emb.values()), model_name=model_name, cuda=args.cuda, batch_size=args.batch_size, query_flag=True)

    doc_embedding_dict = embed_func(all_text_list=list(doc_to_emb.values()), model_name=model_name, cuda=args.cuda, batch_size=args.batch_size, query_flag=False)

    result = {}
    for k in query_embedding_dict:
        result[emb2query[k]] = query_embedding_dict[k]

    for k in doc_embedding_dict:
        result[emb2doc[k]] = doc_embedding_dict[k]

    print(f"-Writing results to {output_path}/{exp_name}.pickle")

    with open(f"{output_path}/{exp_name}.pickle", "wb") as f:
        pickle.dump(result, f)

    

    
