from src.tokenizer import get_token_counts, get_tokenizer
import os


def fixed_token_chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:
    chunks = []
    max_tokens = config["fixed_chunk_size"]
    for emb in config["embedding"]:
        provider = emb["provider"]
        # Get model from the provider-specific section
        if provider == "huggingface":
            model = config["huggingface"][0]["model"]
        elif provider == "openai":
            model = config["openai"][0]["model"]
        else:
            print(f"Unsupported provider: {provider}")
            continue

        tokenizer = get_tokenizer(model, provider)
        
        # Get token IDs based on provider
        if provider == "huggingface":
            all_token_ids = tokenizer.encode(text, add_special_tokens=False)
        elif provider == "openai":
            all_token_ids = tokenizer.encode(text)
        else:
            continue

        id_chunks = [
            all_token_ids[i : i + max_tokens]
            for i in range(0, len(all_token_ids), max_tokens)
        ]
        char_start = 0
        char_end = 0
        for idx, token_id_list in enumerate(id_chunks):
            # Decode based on provider
            if provider == "huggingface":
                chunk_text = tokenizer.decode(token_id_list, skip_special_tokens=True)
            else:  # OpenAI
                chunk_text = tokenizer.decode(token_id_list)
                
            char_end = char_start + len(chunk_text)
            doc = os.path.splitext(doc_id)[0]
            chunk_id = f"{doc}_ft_{idx + 1}"
            chunk_tokens = {model: len(token_id_list)}
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
                "strategy": "fixed_token",
                "source": doc_id,
                "tokens": chunk_tokens,
            })
            char_start = char_end + 1
    return chunks