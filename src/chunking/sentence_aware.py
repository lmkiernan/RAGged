from src.tokenizer import get_token_counts

def sentence_aware_chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:

    token_counts = get_token_counts(text, model_provider_map)
    print(token_counts)


    return []