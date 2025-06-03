from src.tokenizer import get_token_counts, get_tokenizer
import os


def fixed_token_chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:
    chunks = []
    max_tokens = config["fixed_chunk_size"]
    for emb in config["embedding"]:
        provider = emb["provider"]
        tokenizer = get_tokenizer(emb["model"], provider)
        if provider == "huggingface":
            all_token_ids = tokenizer.encode(text, add_special_tokens=False)
            model = "BAAI/bge-large-en"
        else:
            print("wrong provider")
           # once openai added: all_token_ids = tokenizer.encode(text)
        id_chunks = [
            all_token_ids[i : i + max_tokens]
            for i in range(0, len(all_token_ids), max_tokens)
        ]
        char_start = 0
        char_end = 0
        for idx, token_id_list in enumerate(id_chunks):
            chunk_text = tokenizer.decode(token_id_list, skip_special_tokens = False)
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