import json
import time

import cohere
from cohere import Client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI

def load_api_keys(file_path, key_name):
    with open(file_path, 'r') as file:
        api_keys = json.load(file)
    return api_keys.get(key_name)

# replace with your API key file, or specify keys below directly
api_filename = 'api_keys.txt' 
openai_api_key = load_api_keys(api_filename, 'openai_key') # can replace with API key directly 
openai_client = OpenAI(api_key=openai_api_key)
mistral_api_key = load_api_keys(api_filename, 'mistral_key')
mistral_client = MistralClient(api_key=mistral_api_key)
cohere_api_key = load_api_keys(api_filename, 'cohere_key')
cohere_client = cohere.Client(api_key=cohere_api_key)

def retry(func, *args, max_retries=4, wait_time=2, **kwargs):
    """Generic retry mechanism that attempts to run a function multiple times."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                print("All attempts failed.")
                return None

def compute_embeddings_openai(text, model='text-embedding-3-small'):
    response = openai_client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def compute_embeddings_mistral(text, model='mistral-embed'):
    response = mistral_client.embeddings(
        model=model,
        input=text
    )
    return response.data[0].embedding

def compute_embeddings_cohere(text, model='embed-english-v3.0', input_type='search_document'):
    response = cohere_client.embed(
        model=model,
        texts=[text],
        input_type=input_type
    )
    return response.embeddings[0]  

def compute_embeddings(text, model, input_type='search_document'):
    """
    Compute text embeddings using the specified model.
    Currently supports OpenAI, Cohere, and Mistral models.
    """
    embedding_functions = {
        'embed-english-v3.0': lambda: compute_embeddings_cohere(text, model=model, input_type=input_type),
        'mistral-embed': lambda: compute_embeddings_mistral(text, model=model),
        'text-embedding-3-small': lambda: compute_embeddings_openai(text, model=model),
        'text-embedding-3-large': lambda: compute_embeddings_openai(text, model=model)
    }
    
    if model not in embedding_functions:
        raise ValueError(f"Unsupported model: {model}")
    
    return retry(embedding_functions[model])

def get_completion(prompt, model_org, model_name, temp= 0.0):
    """
    Get generation from specified model, with specified temperature. 
    Currently supports OpenAI, Cohere, and Mistral models.
    """
    model_functions = {
        'cohere': lambda: cohere_client.chat(
            model=model_name,
            message=prompt,
            temperature=temp
        ).text,
        
        'mistral': lambda: mistral_client.chat(
            model=model_name,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=temp
        ).choices[0].message.content,
        
        'openai': lambda: openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp
        ).choices[0].message.content
    }
    
    for org, func in model_functions.items():
        if org in model_org:
            return retry(func)
            
    raise ValueError(f"Unknown model organization: {model_org}")