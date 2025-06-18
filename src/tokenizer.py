import tiktoken
from transformers import AutoTokenizer

def get_tokenizer(model_name: str, provider: str):
    if provider.lower() == "huggingface":
        return AutoTokenizer.from_pretrained(model_name)
    elif provider.lower() == "openai":
        return tiktoken.encoding_for_model(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
def count_tokens(text: str, tokenizer, provider: str) -> int:
    if provider.lower() == "huggingface":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    elif provider.lower() == "openai":
        tokens = tokenizer.encode(text)
        return len(tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
def get_token_counts(text: str, provider: str, model_name: str) -> int:
    tokenizer = get_tokenizer(model_name, provider)
    token_counts = count_tokens(text, tokenizer, provider)
    return token_counts
