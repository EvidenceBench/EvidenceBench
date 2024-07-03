import numpy as np
import os
from datetime import datetime
import pytz
import csv
import pickle
import re

def parse_hyena_output_second(prompt, text):
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    
    # Define the pattern to match the final list of indices
    pattern = r'\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    
    if matches:
        last_match = matches[-1]  # Get the last match
        parts = last_match.split(',')
        numbers = []
        for part in parts:
            part = part.strip()
            if re.match(r'^\d+$', part):
                numbers.append(int(part))
        return numbers
    
    return []

def parse_hyena_output(prompt, text):
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    text = text.replace("index", "")
    pattern = r'DECISION:\s*[-*]*\s*\[([0-9, \-]+)\]'
    
    matches = re.findall(pattern, text)
    if matches:
        last_match = matches[-1]
        parts = last_match.split(',')
        numbers = []
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(part))
        return numbers
    else:
        return parse_hyena_output_second(prompt, text)

def parse_claude_output(prompt, text):
    # This regex looks for the word 'DECISION' followed by a colon and brackets containing numbers separated by commas
    pattern = r'DECISION:.*?\[([0-9, ]+)\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        numbers = match.group(1).split(',')
        numbers = [int(num.strip()) for num in numbers]
        return numbers
    return []


def parse_llama_output_second(prompt, text):
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    pattern = r'DECISION:\s*[-*]*\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    if matches:
        last_match = matches[-1]
        parts = last_match.split(',')
        numbers = []
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                numbers.extend(range(start, end + 1))
            elif re.match(r'^\d+$', part):
                numbers.append(int(part))
        return numbers
    return []

def parse_llama_output(prompt, text):
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    pattern = r'DECISION:\s*[-*]*\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    if matches:
        last_match = matches[-1]
        parts = last_match.split(',')
        numbers = []
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                numbers.extend(range(start, end + 1))
            elif re.match(r'^\d+$', part):
                numbers.append(int(part))
        return numbers
    return []
    # text = text.replace("```python", "")
    # text = text.replace("```", "")
    # text = text.replace("*", "")
    # text = text.replace("index", "")
    # pattern = r'DECISION:\s*[-*]*\s*\[([0-9, \-]+)\]'

    # # Find all matches of the pattern
    # matches = re.findall(pattern, text)

    # # Check if there are any matches and extract numbers from the last match
    # if matches:
    #     # Get the last match
    #     last_match = matches[-1]
    #     parts = last_match.split(',')
    #     numbers = []
    #     for part in parts:
    #         part = part.strip()
    #         if '-' in part:
    #             start, end = map(int, part.split('-'))
    #             numbers.extend(range(start, end + 1))
    #         else:
    #             numbers.append(int(part))
    #     return numbers
    # else:
    #     return parse_llama_output_second(prompt, text)

def parse_gpt_output(prompt, text):
    # This regex looks for the word 'DECISION' followed by a colon and brackets containing numbers separated by commas
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    text = text.replace("index", "")
    pattern = r'DECISION:\s*[-*]*\s*\[([0-9, ]+)\]'

    # Find all matches of the pattern
    matches = re.findall(pattern, text)

    # Check if there are any matches and extract numbers from the last match
    if matches:
        # Get the last match
        last_match = matches[-1]
        numbers = [int(num.strip()) for num in last_match.split(',')]
        return numbers
    return []

def parse_gemini_output(prompt, text):
    text = text.replace("Revised Selection", "DECISION").replace("Decision", "DECISION")
    pattern = r'(?:\*\*|##)?\s*DECISION:\s*[-*]*\s*\[([0-9, ]+)\]'

    # Find all matches of the pattern
    matches = re.findall(pattern, text)

    # Check if there are any matches and extract numbers from the last match
    if matches:
        # Get the last match
        last_match = matches[-1]

        if last_match.strip() == "":
            return []
        # Split the last match into individual items (numbers or ranges)
        items = last_match.split(',')
        numbers = []

        for item in items:
            item = item.strip()
            if '-' in item:
                # Expand the range
                start, end = map(int, item.split('-'))
                numbers.extend(range(start, end + 1))
            else:
                # Convert single number to integer and add to the list
                numbers.append(int(item))
        return numbers
    return []

def parse_gemini_output_SBS(prompt, text):
    text = text.replace("Revised Selection", "DECISION").replace("Decision", "DECISION")
    # This regex looks for the word 'DECISION' followed by a colon and brackets containing numbers separated by commas
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    text = text.replace("index", "")
    pattern = r'DECISION:\s*[-*]*\s*\[([0-9, ]+)\]'

    # Find all matches of the pattern
    matches = re.findall(pattern, text)

    # Check if there are any matches and extract numbers from the last match
    if matches:
        # Get the last match
        last_match = matches[-1]
        if last_match.strip() == "":
            return []
            
        numbers = [int(num.strip()) for num in last_match.split(',')]
        return numbers
    return []


def get_west_coast_time():
    # Define the timezone for US West Coast (Pacific Time)
    pacific_time = pytz.timezone('America/Los_Angeles')

    # Get the current time in Pacific Time
    now = datetime.now(pacific_time)

    # Format the time as mm_dd_hour_minute
    formatted_time = now.strftime('%m_%d_%H_%M')
    
    return formatted_time

def append_row_to_csv(file_path, row_dict):
    # Read the existing CSV file

    file_exists = os.path.exists(file_path)

    if not file_exists:
        column_names = ['Experiment Name', 'Time']

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
        print(f'CSV file {file_path} created with column name: {column_names}')
    
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        # Update the fieldnames with new columns from the dictionary
        new_fieldnames = set(row_dict.keys()) - set(fieldnames)
        fieldnames.extend(new_fieldnames)

        rows = list(reader)

    # Create a new dictionary for the row with missing keys set to empty strings
    new_row = {key: row_dict.get(key, '') for key in fieldnames}

    # Append the new row to the existing rows
    rows.append(new_row)

    # Write the updated rows back to the CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_multi_turn_prompt_to_fix_output(prompt_key, next_round_prompt, init_prompt_path, model_output_path):
    with open(init_prompt_path, 'rb') as f:
        init_prompt_dict = pickle.load(f)
    with open(model_output_path, 'rb') as f:
        output_dict = pickle.load(f)

    init_prompt = init_prompt_dict[prompt_key]
    output = output_dict[prompt_key]

    new_prompt = {}

    new_prompt['conversation'] = []

    new_prompt['conversation'].append({'role': 'user', 'content': init_prompt})
    new_prompt['conversation'].append({'role': 'assistant', 'content': output})
    new_prompt['conversation'].append({'role': 'user', 'content': next_round_prompt})

    return new_prompt
