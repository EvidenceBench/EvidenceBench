from gritlm import GritLM

def embed_by_GritLM(dict_to_emb, instruction, mode):

    def gritlm_instruction(instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    
    model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
    
    documents = [dict_to_emb[k] for k in dict_to_emb]

    if mode not in ['query', 'document']:
        raise ValueError(f"mode must be one of ['query', 'document'], the current value is {mode}")

    if mode == 'query':
        rep = model.encode(documents, instruction=gritlm_instruction(instruction))
    else:
        rep = model.encode(documents, instruction=gritlm_instruction(""))

    result = {}
    for ind, k in enumerate(dict_to_emb):
        result[k] = rep[ind]
    return result