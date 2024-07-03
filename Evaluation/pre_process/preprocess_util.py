import pickle
from tqdm import tqdm
import tiktoken
import numpy as np
from itertools import combinations
import os

def remove_abstract_for_dataset(dataset):
    dataset_no_abs = []
    for point in dataset:
        new_point = {}
        for k in point:
            new_point[k] = point[k]

        new_candidate = []
        new_candidate_class = []
        new_aspect2sent = {}
        new_sent2aspect = {}

        new_sentid2old_sentid = {}
        old_sentid2new_sentid = {}

        count = 0
        for ind, cand in enumerate(point['paper_as_candidate_pool']):
            curr_cand_cat = point['sentence_types_in_candidate_pool'][ind]
            if curr_cand_cat == 'abstract':
                continue

            new_candidate.append(cand)
            new_candidate_class.append(curr_cand_cat)

            new_sentid2old_sentid[count] = ind
            old_sentid2new_sentid[ind] = count
            count += 1

        # print(point['aspect2sent'])

        for k in point['aspect2sent']:
            new_aspect2sent[k] = []
            for i in point['aspect2sent'][k]:
                if i not in old_sentid2new_sentid:
                    continue
                new_aspect2sent[k].append(old_sentid2new_sentid[i])
            

        for k in point['sentence_index2aspects']:
            if k not in old_sentid2new_sentid:
                continue
            new_sent2aspect[old_sentid2new_sentid[k]] = point['sentence_index2aspects'][k]

        new_point['paper_as_candidate_pool'] = new_candidate
        new_point['sentence_types_in_candidate_pool'] = new_candidate_class

        new_point['aspect2sent'] = new_aspect2sent
        new_point['sentence_index2aspects'] = new_sent2aspect

        
        dataset_no_abs.append(new_point)

    return dataset_no_abs
    

        

def create_prompt_section_by_section(dataset, prompt_template, text_ele_limit):

    prompt_id2cand_id = {}
    prompt_dict = {}
    
    for k,point in dataset.items():

        if text_ele_limit == -1:
            curr_limit = point['evidence_retrieval_at_optimal_evaluation']['optimal']
        else: 
            curr_limit = text_ele_limit
        
        curr_hypothesis = point['hypothesis']
        curr_hint = k
        section_by_section = split_by_section(point)
        secid2candid = section_by_section['sec2cand_id']

        # all_tables_cand_id = []
        # all_figures_cand_id = []
        # all_tables_figures_cand_id = []

        # if 'tables_and_figures' in secid2candid:
        #     all_tables_figures_cand_id = secid2candid['tables_and_figures']
        # if 'figures' in secid2candid:
        #     all_figures_cand_id = secid2candid['figures']
        
        for section_name in secid2candid:
            curr_prompt_id2cand_id = {}
            # if section_name in ['tables_and_figures']:
            #     continue
            if len(secid2candid[section_name]) == 0:
                continue

            cand_str = ""
            count = 1

            for cid in secid2candid[section_name]:

                # if section_name == 'tables_and_figures' and point['hint'] == wrong_hint:
                #     pdb.set_trace()

                cand_str += f"\n{count}: {point['paper_as_candidate_pool'][int(cid)]}\n"
                curr_prompt_id2cand_id[count] = cid
                count += 1

                
            curr_prompt = prompt_template.format(hypothesis=curr_hypothesis, cand_pool=cand_str, text_ele_limit=curr_limit)

            if num_tokens_from_string(curr_prompt) > 6000:
                # in this case, the section is too long, we need to split the section from the middle

                full_length = len(secid2candid[section_name])

                sub_1_section_idx = (0, int(full_length*0.325))
                sub_2_section_idx = (int(full_length*0.225), int(full_length*0.55))
                sub_3_section_idx = (int(full_length*0.45), int(full_length*0.775))
                sub_4_section_idx = (int(full_length*0.675), full_length)

                cand_str_1 = ""
                cand_str_2 = ""
                cand_str_3 = ""
                cand_str_4 = ""

                count1 = 0
                count2 = 0
                count3 = 0
                count4 = 0

                curr_prompt_id2cand_id_1 = {}
                curr_prompt_id2cand_id_2 = {}
                curr_prompt_id2cand_id_3 = {}
                curr_prompt_id2cand_id_4 = {}

                for cid in secid2candid[section_name][sub_1_section_idx[0]: sub_1_section_idx[1]]:
                    cand_str_1 += f"\n{count1}: {point['paper_as_candidate_pool'][int(cid)]}\n"
                    curr_prompt_id2cand_id_1[count1] = cid
                    count1 += 1

                for cid in secid2candid[section_name][sub_2_section_idx[0]: sub_2_section_idx[1]]:
                    cand_str_2 += f"\n{count2}: {point['paper_as_candidate_pool'][int(cid)]}\n"
                    curr_prompt_id2cand_id_2[count2] = cid
                    count2 += 1

                for cid in secid2candid[section_name][sub_3_section_idx[0]: sub_3_section_idx[1]]:
                    cand_str_3 += f"\n{count3}: {point['paper_as_candidate_pool'][int(cid)]}\n"
                    curr_prompt_id2cand_id_3[count3] = cid
                    count3 += 1

                for cid in secid2candid[section_name][sub_4_section_idx[0]: sub_4_section_idx[1]]:
                    cand_str_4 += f"\n{count4}: {point['paper_as_candidate_pool'][int(cid)]}\n"
                    curr_prompt_id2cand_id_4[count4] = cid
                    count4 += 1


                prompt_dict[(curr_hint, f"{section_name}_{1}")] = prompt_template.format(hypothesis=curr_hypothesis, cand_pool=cand_str_1, text_ele_limit=curr_limit)
                prompt_dict[(curr_hint, f"{section_name}_{2}")] = prompt_template.format(hypothesis=curr_hypothesis, cand_pool=cand_str_2, text_ele_limit=curr_limit)
                prompt_dict[(curr_hint, f"{section_name}_{3}")] = prompt_template.format(hypothesis=curr_hypothesis, cand_pool=cand_str_3, text_ele_limit=curr_limit)
                prompt_dict[(curr_hint, f"{section_name}_{4}")] = prompt_template.format(hypothesis=curr_hypothesis, cand_pool=cand_str_4, text_ele_limit=curr_limit)

                
                prompt_id2cand_id[(curr_hint, f"{section_name}_{1}")] = curr_prompt_id2cand_id_1
                prompt_id2cand_id[(curr_hint, f"{section_name}_{2}")] = curr_prompt_id2cand_id_2
                prompt_id2cand_id[(curr_hint, f"{section_name}_{3}")] = curr_prompt_id2cand_id_3
                prompt_id2cand_id[(curr_hint, f"{section_name}_{4}")] = curr_prompt_id2cand_id_4

                continue



            prompt_dict[(curr_hint, section_name)] = curr_prompt
            prompt_id2cand_id[(curr_hint, section_name)] = curr_prompt_id2cand_id


    return prompt_dict, prompt_id2cand_id

def create_multi_turn_prompt_to_fix_output(prompt_key, new_prompt, init_prompt_path, model_output_path):
    with open(init_prompt_path, 'rb') as f:
        init_prompt_dict = pickle.load(f)
    with open(model_output_path, 'rb') as f:
        output_dict = pickle.load(f)

    init_prompt = init_prompt_dict[prompt_key]
    output = output_dict[prompt_key]

    new_prompt_dict = []

    new_prompt_dict['conversation'] = []

    new_prompt_dict['conversation'].append({'role': 'user', 'content': init_prompt})
    new_prompt_dict['conversation'].append({'role': 'assistant', 'content': output})
    new_prompt_dict['conversation'].append({'role': 'user', 'content': new_prompt})

    return new_prompt_dict

def list_files_abs_path(directory):
    directory = os.path.abspath(directory)  # Ensure the path is absolute
    file_paths = []  # List to store the absolute paths of files

    # Get all files directly under the specified directory, excluding hidden files
    for file in os.listdir(directory):
        if file.startswith('.'):
            continue  # Skip hidden files
        abs_path = os.path.join(directory, file)
        if os.path.isfile(abs_path):  # Ensure it's a file, not a directory
            file_paths.append(abs_path)

    # Sort the file paths by the number in the filename
    file_paths = sorted(file_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return file_paths

def find_img_path_for_point(point):
    point_id = f"{point['id_type']}_{point['id']}"
    file_path_list = list_files_abs_path(f"/data/Elsevier/Evaluation/generation/images/{point_id}/")

    return file_path_list


def num_tokens_from_string(string, encoding_name='gpt-4-turbo'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def find_minimal_covering_keys(data):
    # Collect all unique values that need to be covered
    all_values = set().union(*data.values())
    
    # Find the minimal covering sets
    smallest_cover = None
    smallest_size = float('inf')
    
    # Iterate over all possible combinations of keys
    for size in range(1, len(data) + 1):
        for keys_subset in combinations(data.keys(), size):
            # Compute the union of values for the current subset of keys
            current_union = set().union(*(data[key] for key in keys_subset))
            
            # Check if current union covers all values
            if current_union == all_values:
                if size < smallest_size:
                    smallest_size = size
                    smallest_cover = [set(keys_subset)]
                elif size == smallest_size:
                    smallest_cover.append(set(keys_subset))
    
    return smallest_cover

def split_by_section(point):
    # The point should be a dict, for each dict in the list, it should have the following keys:
    # 'paper_as_candidate_pool': list of strings, each string is a candidate sentence
    # 'sentence_types_in_candidate_pool': list of strings, each string is the class of the corresponding candidate sentence
    # 'sentence_index2aspects': dict of lists, each list is the corresponding aspects of the sentence in the candidate pool with the index
    
    # Output: (split_by_section_dataset, section2asp)

    sec2cand_id = {}
    sec2cand_id['abstracts'] = []
    sec2cand_id['tables_and_figures'] = []
    # sec2cand_id['figures'] = []

    sec2asp_id = {}
    asp_id2sec = {}

    sec_count = 0
    curr_sec_name = f"section_{sec_count}"
    curr_sec_content_list = []

    for ind, content in enumerate(point['paper_as_candidate_pool']):
        str_ind = str(ind)
        if point['sentence_types_in_candidate_pool'][ind] != 'section_name':
            if point['sentence_types_in_candidate_pool'][ind] == 'abstract':
                sec2cand_id['abstracts'].append(str_ind)
            elif point['sentence_types_in_candidate_pool'][ind] == 'table_caption' or point['sentence_types_in_candidate_pool'][ind] == 'table_row' or point['sentence_types_in_candidate_pool'][ind]=='figure':
                sec2cand_id['tables_and_figures'].append(str_ind)
            elif point['sentence_types_in_candidate_pool'][ind] == 'normal_paragraph':
                curr_sec_content_list.append(str_ind)
            # elif point['sentence_types_in_candidate_pool'][ind] == 'figure':
            #     sec2cand_id['figures'].append(ind)
            

        else:
            # this is the case a new section name is read
            sec2cand_id[curr_sec_name] = curr_sec_content_list
            sec_count += 1
            curr_sec_name = f"section_{sec_count}"
            curr_sec_content_list = [str(ind)]

    sec2cand_id[curr_sec_name] = curr_sec_content_list

    for section_name in sec2cand_id:
        curr_sec_asps = []
        for sent_ind in sec2cand_id[section_name]:
            curr_sec_asps += point['sentence_index2aspects'][sent_ind]
        curr_sec_asps = set(curr_sec_asps)
        sec2asp_id[section_name] = curr_sec_asps
    
    for section_name in sec2asp_id:
        for asp in sec2asp_id[section_name]:
            if asp not in asp_id2sec:
                asp_id2sec[asp] = [section_name]
            else:
                asp_id2sec[asp].append(section_name)

    return {'sec2cand_id': sec2cand_id, 'sec2asp_id': sec2asp_id, 'asp_id2sec': asp_id2sec}

def remove_empty_sections(ret):

    sec2cand_id, sec2asp_id, asp_id2sec = ret['sec2cand_id'], ret['sec2asp_id'], ret['asp_id2sec']

    new_sec2cand_id = {}
    new_sec2asp_id = {}
    
    
    for sec in sec2asp_id:
        if len(sec2asp_id[sec]) != 0:
            new_sec2asp_id[sec] = sec2asp_id[sec]
            new_sec2cand_id[sec] = sec2cand_id[sec]

    return {'sec2cand_id': new_sec2cand_id, 'sec2asp_id': new_sec2asp_id, 'asp_id2sec': asp_id2sec} 
            


def rank_section_by_priority(ret, point):

    sec2cand_id, sec2asp_id, asp_id2sec = ret['sec2cand_id'], ret['sec2asp_id'], ret['asp_id2sec']

    min_section_set_cover_all_asp = find_minimal_covering_keys(sec2asp_id)[0]

    section_to_remove = [sec for sec in sec2cand_id if sec not in min_section_set_cover_all_asp ]

    cand_length = {}
    for ind, cand in enumerate(point['paper_as_candidate_pool']):
        cand_length[ind] = num_tokens_from_string(cand)

    sec_length = {}
    for sec in section_to_remove:
        sec_length[sec] = 0
        for cand_id in sec2cand_id[sec]:
            sec_length[sec] += cand_length[cand_id]

    sorted_data = {k: v for k, v in sorted(sec_length.items(), key=lambda item: item[1], reverse=True)}

    return list(sorted_data.keys())


def tok_length_of_cand_pool(cand_pool_ids, point):
    ret = 0
    for cand_id in cand_pool_ids:
        ret += num_tokens_from_string(point['paper_as_candidate_pool'][cand_id])
    return ret


def reduce_cand_pool(point, max_token=3000):

    cand_pool_ids = list(point['sentence_index2aspects'].keys())

    tok_length = tok_length_of_cand_pool(cand_pool_ids, point)

    if tok_length <= max_token:
        return list(point['sentence_index2aspects'].keys())

    splitted_by_section = split_by_section(point)
    remaining = remove_empty_sections(splitted_by_section)

    new_cand_ids = []

    for sec in remaining['sec2cand_id']:
        new_cand_ids += remaining['sec2cand_id'][sec]

    new_cand_ids = list(sorted(new_cand_ids))

    tok_length = tok_length_of_cand_pool(new_cand_ids, point)

    if tok_length <= max_token:
        return new_cand_ids

    sec_to_remove = rank_section_by_priority(remaining, point)

    remaining_cand_ids = new_cand_ids

    while len(sec_to_remove) > 0 and tok_length > max_token:

        removing_sec = sec_to_remove.pop(0)
        new_remaining_cand_ids = [cand_id for cand_id in remaining_cand_ids if cand_id not in remaining['sec2cand_id'][removing_sec]]
        remaining_cand_ids = new_remaining_cand_ids

        tok_length = tok_length_of_cand_pool(remaining_cand_ids, point)

    if tok_length <= max_token:
        return remaining_cand_ids
    


    needed_sent = []
    for cand_id in remaining_cand_ids:
        
        if len(point['sentence_index2aspects'][cand_id]) != 0:
            needed_sent.append(cand_id)

    new_sent2asp = {}
    for cand_id in needed_sent:
        new_sent2asp[cand_id] = set(point['sentence_index2aspects'][cand_id])

    min_sent_cover = list(find_minimal_covering_keys(new_sent2asp)[0])

    useless_sents = [cand_id for cand_id in remaining_cand_ids if cand_id not in min_sent_cover]

    cand_length = {}
    for cand_id in useless_sents:
        cand_length[cand_id] = num_tokens_from_string(point['paper_as_candidate_pool'][cand_id])

    sorted_dict = {k: v for k, v in sorted(cand_length.items(), key=lambda item: item[1], reverse=True)}
    assert len(sorted_dict) == len(useless_sents)
    cand_id_to_remove = list(sorted_dict.keys())

    while len(cand_id_to_remove) > 0 and tok_length > max_token: 
        removing_cand_id = cand_id_to_remove.pop(0)
        tok_length -= cand_length[removing_cand_id]
    if tok_length <= max_token:
        return list(sorted(cand_id_to_remove + needed_sent))
    else:
        return None
    

def shorten_candidate_pool_for_dataset(dataset, max_token=3000):
    prob_point_ind = []
    shorter_cand_pool_full = {}

    pbar = tqdm(total=len(dataset))

    for ind, point in enumerate(dataset):
        shorter_cand_pool = reduce_cand_pool(point, max_token)
        if shorter_cand_pool is None:
            prob_point_ind.append(ind)
            shorter_cand_pool_full[ind] = None
        else:
            shorter_cand_pool_full[ind] = shorter_cand_pool

        pbar.update(1)

    print(f"The following points cannot have shorten candidate pool: {prob_point_ind}")
    return shorter_cand_pool_full
        
        