import pickle
from tqdm import tqdm

import pulp

def min_key_cover_ilp(input_dict):
    # All numbers we need to cover
    all_numbers = set(chain.from_iterable(input_dict.values()))
    
    # Create the ILP problem
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    
    # Create a binary variable for each key
    key_vars = pulp.LpVariable.dicts("Key", input_dict.keys(), cat='Binary')
    
    # Objective function: minimize the sum of the selected keys
    prob += pulp.lpSum([key_vars[k] for k in input_dict.keys()])
    
    # Add constraints to ensure all numbers are covered
    for number in all_numbers:
        prob += pulp.lpSum([key_vars[k] for k in input_dict.keys() if number in input_dict[k]]) >= 1, f"Cover_{number}"
    
    # Solve the problem
    prob.solve()
    
    # Extract the selected keys
    selected_keys = [k for k in input_dict.keys() if pulp.value(key_vars[k]) == 1]
    
    return set(selected_keys)

with open("../dataset/ARIADNE_300_test_set_may20_annotated.pickle", 'rb') as f:
    dataset = pickle.load(f)

optimal_k_list = []
full_list = []
for point in tqdm(dataset):
    # print(len(point['sent2aspect']))
    curr_sent2aspect = point['sent2aspect']
    ret = min_key_cover(curr_sent2aspect)
    full_list.append(ret)
    optimal_k = len(ret)
    optimal_k_list.append(optimal_k)

print(optimal_k_list)
with open("./optimal_k_list.pickle", 'wb') as f:
    pickle.dump(f)