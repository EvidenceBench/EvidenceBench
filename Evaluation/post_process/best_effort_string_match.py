import re
import editdistance

def extract_text_after_numbers(text):
    # Regular expression pattern to match text following a number.
    # This pattern ensures that we capture all text until we hit a line that starts with a number and period
    pattern = r'^\d+[:.]\s(.*(?:\n|$))'
    
    matches = re.findall(pattern, text, re.MULTILINE)
    return [match.strip() for match in matches]

def find_most_similar_substring(text, search_text):
    len1 = len(search_text)
    len2 = len(text)
    min_distance = float('inf')
    best_match = None
    best_start = -1
    best_end = -1

    # Loop through all possible starting indices in text
    for start in range(len2 - len1 + 1):  # Adjust the range to ensure substrings match the length of search_text
        end = start + len1 - 1  # Calculate the ending index to ensure the substring has the same length as search_text
        substring = text[start:end+1]
        distance = editdistance.eval(search_text, substring)
        if distance < min_distance:
            min_distance = distance
            best_match = substring
            best_start = start
            best_end = end

    return best_match, best_start, best_end, min_distance

def find_number_at_line_start_before_substring(sub_str, full_str):
    # Compile a regular expression that looks for a line starting with up to 3 digits followed immediately by a '.' or ':'
    pattern = r'(?m)^(?:\d{1,4})[:\.]'
    
    # Find all matches of the pattern in the full string
    sub_str = sub_str.lower()
    full_str = full_str.lower()
    matches = list(re.finditer(pattern, full_str))
    
    if not matches:
        return None
    
    # Locate the position of the sub_str in the full_str
    sub_str_index = full_str.find(sub_str)
    
    # Find the closest pattern before the sub_str_index
    closest_match = None
    min_distance = float('inf')
    
    for match in matches:
        # Calculate the distance from the end of the match to the start of sub_str
        distance = sub_str_index - match.end()
        
        if distance >= 0 and distance < min_distance:
            min_distance = distance
            closest_match = match.group()
    
    return closest_match[:-1] if closest_match else None


def find_number_preceding_text(text, search_text):
    # Escape special characters in the search text to use in regex
    escaped_search_text = re.escape(search_text)
    
    # Regular expression pattern to find the number before the specified text
    # Ensuring that the search text is exactly matched and followed by a non-word character or end of line
    exact_pattern = r'^(\d+)\:\s+' + escaped_search_text + r'(?=[.\n])'
    
    # Try finding an exact match first
    exact_matches = re.findall(exact_pattern, text, re.MULTILINE | re.IGNORECASE)
    if exact_matches:
        return exact_matches

    partial_match_output = find_most_similar_substring(text, search_text)
    if partial_match_output[3] >= 0.3:
        return []
    else:
        partial_match = find_number_at_line_start_before_substring(search_text, text)
        if partial_match:
            return [partial_match]
        else:
            return []

def find_matched_list(prompt, model_output):
    index_list = []
    found_text_list = extract_text_after_numbers(model_output)
    for text in found_text_list:
        index_list += find_number_preceding_text(prompt, text)
    return [int(k) for k in list(set(index_list))]