import numpy as np
from transformers import AutoTokenizer


max_token_map = {
    "text-davinci-003": 4096,
    "text-curie-001": 2048
}


def num_tokens_from_string(string: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids: np.ndarray = tokenizer.encode(string, return_tensors='np')
    input_ids = input_ids.squeeze(axis=0)
    return input_ids.shape[0]


def calculate_max_token(model: str, prompt: str):
    return max_token_map[model] - num_tokens_from_string(prompt) - 2
