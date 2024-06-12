import time
from voyageai import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from tqdm import tqdm
import tiktoken
import numpy as np

api_key = 'REPLACE WITH YOUR VOYAGE AI API KEY'


def num_tokens_from_string(string, encoding_name='gpt-4-turbo'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def embed_texts_with_voyage(all_text_list, max_workers=50,timeout_seconds=120,model_name="voyage-large-2-instruct", batch_size=0, cuda="0", api_key=api_key, query_flag=False):
    # Initialize the voyageai client
    if api_key:
        vo = Client(api_key=api_key)
    else:
        vo = Client()

    embeddings = {}
    current_batch = []
    current_batch_token_count = 0
    start_time = time.time()

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(all_text_list), desc="Embedding texts")

    call_counter = 0
    minute_start_time = time.time()

    input_type = 'document'
    if query_flag:
        input_type='query'

    def embed_batch(batch):
        nonlocal call_counter, minute_start_time
        attempts = 0
        while attempts < 3:  # Retry up to 3 times
            try:
                # Rate limiting logic
                if call_counter >= 100:
                    elapsed_minute = time.time() - minute_start_time
                    if elapsed_minute < 60:
                        sleep_time = 60 - elapsed_minute
                        print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    call_counter = 0
                    minute_start_time = time.time()

                result = vo.embed(batch, model=model_name, input_type=input_type)
                call_counter += 1
                return {t: np.array(result.embeddings[i]) for i, t in enumerate(batch)}
            except Exception as exc:
                print(f"Error embedding batch: {exc}. Retrying in 5 seconds...")
                time.sleep(5)
                attempts += 1
        return {t: None for t in batch}  # Return None for texts that failed after retries

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for text in all_text_list:
            tokens = num_tokens_from_string(text)

            if current_batch_token_count + tokens > 100000 or len(current_batch) >= 127:
                futures.append(executor.submit(embed_batch, current_batch))

                current_batch = []
                current_batch_token_count = 0
                start_time = time.time()  # Reset start time after API call

            current_batch.append(text)
            current_batch_token_count += tokens

        if current_batch:
            futures.append(executor.submit(embed_batch, current_batch))

        for future in as_completed(futures):
            batch_embeddings = future.result()
            embeddings.update(batch_embeddings)
            pbar.update(len(batch_embeddings))

    pbar.close()
    return embeddings