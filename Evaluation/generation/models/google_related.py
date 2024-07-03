import google.generativeai as genai
import threading
import time

available_keys = ["REPLACE WITH YOUR API KEY"]

# Global dictionary to track the usage of each key
key_usage_count = {key: 0 for key in available_keys}

# Lock for making the function thread-safe
lock = threading.Lock()

def get_least_used_key():
    with lock:
        # Find the key with the minimum usage count
        least_used_key = min(key_usage_count, key=key_usage_count.get)
        
        # Update the usage count for the key that is selected
        key_usage_count[least_used_key] += 1
    
    # Return the least used key outside of the locked context
    return least_used_key

# system_instruction = ""

def generate_gemini_response(prompt, model="gemini-1.5-pro-latest", temperature=0, max_tokens = 1024):

    curr_key = get_least_used_key()

    genai.configure(api_key=curr_key)

    time.sleep(1)

    generation_config = {
    "temperature": temperature,
    "top_p": 1,
    "top_k": 0,
    "max_output_tokens": max_tokens,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]

    model = genai.GenerativeModel(model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings)
    
    convo = model.start_chat(history=[])

    convo.send_message(prompt)
    return convo.last.text, (0,0)

def generate_gemini_response_with_image(prompt, model="gemini-1.5-pro-latest", temperature=0, max_tokens = 1024):

    if model not in ["gemini-1.5-pro-latest"]:
        raise ValueError("We are only going to test gemini-1.5-pro's performance with images.")
    
    curr_key = get_least_used_key()
    genai.configure(api_key=curr_key)

    generation_config = {
    "temperature": temperature,
    "top_p": 1,
    "top_k": 0,
    "max_output_tokens": max_tokens,
    }

    safety_settings = []

    model = genai.GenerativeModel(model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings)
    print("Before loading images")
    convo = model.start_chat(history=[{"role": "user", "parts": [genai.upload_file(img_path)]} for img_path in prompt['images']])
    
    convo.send_message(prompt['prompt_text'])
    return convo.last.text, (0,0)


def generate_gemini_response_multi_turn(prompt, model="gemini-1.5-pro-latest", temperature=0, max_tokens = 1024):

    if type(prompt) is not dict or 'conversation' not in prompt.keys():
        raise ValueError("For multiple turn case, the prompt dict provided is not in correct format")

    curr_key = get_least_used_key()
    # print(f"Curr_key: {curr_key}")
    # print(f"Usage: {key_usage_count}")
    
    genai.configure(api_key=curr_key)

    generation_config = {
    "temperature": temperature,
    "top_p": 1,
    "top_k": 0,
    "max_output_tokens": max_tokens,
    }

    safety_settings = []

    model = genai.GenerativeModel(model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings)

    history = []
    for ind, i in enumerate(prompt['conversation']):
        if ind == len(prompt['conversation']) - 1:
            break
        if i['role'] == 'assistant':
            gemini_role = "model"
        elif i['role'] == 'user':
            gemini_role = 'user'
        else:
            raise ValueError("This function only accept role being ['user', 'assistant']")
        history.append({'role': gemini_role, 
                         'parts': [i['content']]
                         })
    
    convo = model.start_chat(history=history)

    assert prompt['conversation'][-1]['role'] == 'user', "The last prompt's role must be user"

    latest_prompt = prompt['conversation'][-1]['content']

    convo.send_message(latest_prompt)
    return convo.last.text, (0,0)
