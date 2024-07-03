from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pdb
import numpy as np

openai_api_key = "REPLACE WITH YOUR API KEY" 

def openai_embedding(text, model):
    # Note that "prompt" is not used to generate anything. This function only finds the embedding of the prompt

    if model not in ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']:
        raise ValueError(f"INVALID MODEL NAME: {model}")
    client = OpenAI(api_key = openai_api_key)
    response = client.embeddings.create(
    input=text,
    model=model
    )

    return response.data[0].embedding


def call_until_timeout(func, timeout_seconds,delay=1,**kwargs):
    """
    Calls the specified function until it succeeds or the timeout is reached.

    Args:
    - func: The function to call.
    - timeout_seconds: Total time allowed for retries in seconds.
    - delay: Delay between retries in seconds.
    """
    start_time = time.time()  # Record the start time
    while True:
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time > timeout_seconds:
            print("Timeout reached, stopping attempts.")
            return "**********NO OUTPUT**********"
        try:
            # Try to call the function
            # pdb.set_trace()
            result = func(**kwargs)

            return result
        except Exception as e:
            print(f"Function call failed with error: {e}")
            remaining_time = timeout_seconds - elapsed_time
            if remaining_time <= 0:
                print(f"Timeout reached, stopping attempts.")
                return "**********NO OUTPUT**********"
            print(f"Retrying in {min(delay, remaining_time)} seconds...")
            # Wait for either the specified delay or the remaining time, whichever is shorter
            time.sleep(min(delay, remaining_time))

def emb_text_in_parallel_openai(all_text_list, max_workers=250,timeout_seconds=120,model_name="text-embedding-3-small",batch_size=0, cuda="0", query_flag=False):

    # Note that batch_size and cuda are not used in this function. They are only used in the local embedding function.

    """
    Executes calls to `call_until_timeout` in parallel using threading, for each prompt in the prompts_dict.
    
    :param prompts_dict: A dictionary where keys are identifiers and values are prompts.
    :param max_workers: The maximum number of threads to use.
    """

    text_list = all_text_list
    pbar = tqdm(total=len(text_list))
    emebedding_result = {}
    # Define the fixed parameters for the call_until_timeout function.

    if model_name in ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']:
        func = openai_embedding
    else:
        raise ValueError(f"INVALID MODEL NAME: {model_name}")

    fixed_params = {
        "func":func,
        "timeout_seconds": timeout_seconds,
        "model": model_name
    }
    
    # Use ThreadPoolExecutor to run tasks in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_text = {}

        for text in text_list:

            future = executor.submit(call_until_timeout, **fixed_params, text=text)
            future_to_text[future] = text

        
        # Process the results as they complete.
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                result = future.result()
                emebedding_result[text] = np.array(result)
                pbar.update(1)
            except Exception as exc:
                print(f"generated an exception: {exc}")
    pbar.close()
    return emebedding_result