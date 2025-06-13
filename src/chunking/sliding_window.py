from src.tokenizer import get_token_counts, get_tokenizer
import os

def sliding_window_chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:
    max_tokens = config["fixed_chunk_size"]
    overlap = config["overlap"]
    stride = max_tokens - overlap
    if stride <= 0:
        raise ValueError(f"Overlap ({overlap}) must be smaller than chunk_size ({max_tokens}).")
    
    chunks = []
    for emb in config["embedding"]:
        provider = emb["provider"]
        model = emb.get("model", "BAAI/bge-large-en") if provider == "huggingface" else "text-embedding-3-small"
        
        if model not in model_provider_map:
            print(f"Skipping {model} as it's not in model_provider_map")
            continue
            
        tokenizer = get_tokenizer(model, provider)
        if provider == "huggingface":
            all_token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            print("wrong provider")
            # once openai added: all_token_ids = tokenizer.encode(text)
            continue

        total_tokens = len(all_token_ids)
        
        chunks_of_ids = []

        start_idx = 0
        while start_idx < total_tokens:
            end_idx = start_idx + max_tokens
            if end_idx > total_tokens:
                end_idx = total_tokens
            chunks_of_ids.append(all_token_ids[start_idx:end_idx])
            start_idx += stride

        char_start = 0
        for idx, token_id_list in enumerate(chunks_of_ids):
            if provider == "openai":
                chunk_text = tokenizer.decode(token_id_list)
            else:
                chunk_text = tokenizer.decode(token_id_list, skip_special_tokens=True)
            char_end = char_start + len(chunk_text)
            doc = os.path.splitext(doc_id)[0]
            chunk_id = f"{doc}_sw_{idx + 1}"
            chunk_tokens = {model: len(token_id_list)}
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
                "source": doc_id,
                "strategy": "sliding_window",
                "tokens": chunk_tokens
            })
            char_start = char_end + 1

    return chunks