import pickle
import torch
import os
from math import *
import re
import json
import sys
import numpy as np
from itertools import chain
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
import torch.nn.functional as F
import pdb
import argparse
from gritlm import GritLM


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# The function to load the model and tokenizer. if model_path is None, then the pretrained model is loaded
def load_model(model_name, cuda = None, model_path=None):
    """
    loading models/tokenizers based on model_name, also based on cuda option specify whether DataParallel.
    Input: cuda option is a string, e.g. "1,3,5" specify cuda1, cuda3, and cuda5 will be used, store parameters on cuda1. 
    """
    if model_name in ["e5"]:
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        if model_path is None:
            model = AutoModel.from_pretrained('intfloat/e5-large-v2')
        else:
            model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth", map_location='cpu')
            try:
                model = model_parallel.module
            except:
                model = model_parallel
        model.eval()

    elif model_name == "e5_mistral":
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model.eval()

    elif model_name == "simcse":
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        if model_path is None:
            model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        else:
            try:
                model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth", map_location='cpu')
                model = model_parallel.module
            except:
                model_parallel = torch.load(f"../SimCSE/steel_simcse/{model_path}.pth", map_location='cpu')
                model = model_parallel.module
        model.eval()
    else:
        raise ValueError("Model Not Supported")
        
    if cuda!= "cpu":
        torch.cuda.set_device(int(cuda.split(",")[0]))
        
        model.to("cuda")
        cuda_list = cuda.split(",")
        cuda_num = len(cuda_list)
        if cuda_num>1:
            model = torch.nn.DataParallel(model, device_ids = [int(idx) for idx in cuda_list])
    else:
        print("Running model on CPU...")
    num_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} model parameters: {int(num_params/1000000)} millions.")
    return model, tokenizer


def get_embedding(model_name, model, tokenizer, text, cuda= "cpu", batch_size= 100):
    """
    Input model: loaded model
          tokenizer: associated tokenizer
          text: a list of strings, each string is either a query or an abstract
          cuda: in the format of "0,1,6,7" or "0", by default, cpu option is used
          batch_size: if not specified, then an optimal batch_size is found by system, else, 
                       the user specified batch_size is used, may run into OOM error.
    Return:  the embedding dictionary, where the key is a string (e.g. an abstract, query/subquery), and the value
             is np.ndarray of the vector, usually 1 or 2 dimensions. 
    """
    if cuda != "cpu":
        if model_name in ["ot_aspire", "ts_aspire", "sentbert", "ance"]: 
            cuda = cuda.split(",")[0]
        cuda_list = cuda.split(",")
        cuda_num = len(cuda_list)
        batch_size = batch_size
        length = ceil(len(text)/batch_size)
    else:
        batch_size = batch_size
    ret = {}  
    length = ceil(len(text)/batch_size)    
    for i in tqdm(range(length)):
        curr_batch = text[i*batch_size:(i+1)*batch_size]
        curr_batch_cleaned = [t for t in curr_batch]
        if model_name == "ance": 
            inputs = tokenization(model_name, tokenizer, curr_batch_cleaned)
        else:
            inputs = tokenizer(curr_batch_cleaned, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        embedding = encoding(model_name, model, inputs, cuda)
        for t, v in zip(curr_batch, embedding):
            ret[t] = v
    return ret


def tokenization(model_name, tokenizer, text):
    '''
    Different tokenization procedures based on different models.
    
    Input: text as list of strings, if cpu option then list has length 1.
    Return: tokenized inputs, could be dictionary for BERT models. 
    '''
    inputs = tokenizer(text, padding=True, truncation=True, max_length=1000, return_tensors="pt")
    return inputs 


def encoding(model_name, model, inputs, cuda):
    '''
    Different encoding procedures based on different models. 
    Input: inputs are tokenized inputs in specific form
    Return: a numpy ndarray embedding on cpu. 
    
    '''
    if cuda != "cpu":
        device = "cuda"
    else:
        device = "cpu"
    with torch.no_grad():
        if model_name in ["e5", "e5v3"]:
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            new_batch_dict={}
            new_batch_dict["input_ids"] = input_ids
            new_batch_dict["token_type_ids"] = token_type_ids
            new_batch_dict["attention_mask"] = attention_mask

            outputs = model(**new_batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
   
            output = embeddings.detach().cpu()

        elif model_name in ["e5_mistral"]:
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            attention_mask = inputs['attention_mask'].to(device)

            new_batch_dict={}
            new_batch_dict["input_ids"] = input_ids
            new_batch_dict["attention_mask"] = attention_mask

            outputs = model(**new_batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
   
            output = embeddings.detach().cpu()

        elif model_name in ["simcse"]:
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            attention_mask = inputs['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask)
            outputs = outputs.pooler_output
            outputs = F.normalize(outputs, p=2, dim=1)
            output = outputs.detach().cpu()
    return output.numpy()

def emb_text_gritlm(text_list, model, instruction):
    output = model.encode(text_list, instruction=instruction)
    emb_dict = {}
    for i in tqdm(range(len(output))):
        emb_dict[text_list[i]] = output[i]
    return emb_dict

def get_embedding_dict_local(model_name, all_text_list, cuda, batch_size, query_flag=False):
    if 'gritlm' in model_name.lower():
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
        if query_flag:
            # gritlm_query_instr = "From a biomedical experiment, find important and representative details that would form the most effective set of evidence relevant to the hypothesis."
            gritlm_query_instr = """From a biomedical paper, find important and representative details about experiment outcomes, results and analyses that would form the most effective set of evidence relevant to the hypothesis."""
            # This instruction is hard coded

            instruction = "<|user|>\n" + gritlm_query_instr + "\n<|embed|>\n"
            # instruction = "<|embed|>\n"
        else:
            instruction = "<|embed|>\n"

        all_emb_dict = emb_text_gritlm(all_text_list, model, instruction)

    else:

        model, tokenizer = load_model(model_name, cuda)
        all_emb_dict = get_embedding(model_name, model, tokenizer, all_text_list, cuda, batch_size)
    return all_emb_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda", type = str, required = True, help = "list of gpus indices")
    parser.add_argument("-model_name", type = str, required = True, help = "pretrained model, such as e5")
    parser.add_argument("-emb_data_path", type = str, required = True, help = "path to text to be embedded")
    parser.add_argument("-batch_size", "--batch_size", type = int, required = True, help = "batch size")
    parser.add_argument('--emb_store_path', default=None, type=str, required=True, help='the path to store the embeddings')

    args = parser.parse_args()
    
    cuda = args.cuda
    model_name = args.model_name
    batch_size = args.batch_size
    emb_data_path = args.emb_data_path
    emb_store_path = args.emb_store_path
    
    with open(emb_data_path, 'rb') as f:
        all_text_list = pickle.load(f)
        
    print(f"Loading model: {model_name}....")
    
    model, tokenizer = load_model(model_name, cuda)
    all_emb_dict = get_embedding(model_name, model, tokenizer, all_text_list, cuda, batch_size)    
