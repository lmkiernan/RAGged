import os
import json
from src.config import load_config
from src.Embedding import OpenAIEmbedder, HFEmbedder
from src.vectorStore import search

def load_api_keys():
    try:
        with open("APIKeys.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("APIKeys.json not found. Please ensure it exists in the root directory.")
    except json.JSONDecodeError:
        raise ValueError("APIKeys.json is not valid JSON.")

def main():
    # 1) Load configuration
    cfg = load_config("config/default.yaml")

    # 2) Load API keys
    api_keys = load_api_keys()

    # 3) Determine which embedder to use (pick the first in the list)
    emb_configs = cfg.get("embedding", [])
    if not emb_configs:
        raise ValueError("No embedding configuration found in config/default.yaml")

    primary = emb_configs[0]
    provider = primary.get("provider").lower()

    # 4) Instantiate the appropriate Embedder
    if provider == "openai":
        api_key = api_keys.get("openai")
        if not api_key:
            raise ValueError("OpenAI API key not found in APIKeys.json")
        model_name = cfg.get("openai")[0].get("model")
        embedder = OpenAIEmbedder(model_name, api_key)
    elif provider in ("huggingface", "hf"):  
        api_key = api_keys.get("huggingface")
        if not api_key:
            raise ValueError("Hugging Face API key not found in APIKeys.json")
        model_name = cfg.get("huggingface")[0].get("model")
        embedder = HFEmbedder(model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    # 5) Load retrieval parameters
    top_k = cfg.get("objectives", {}).get("retrieval_top_k", 5)

    # 6) Prompt user for a query
    query = input("\nEnter a natural-language query: ").strip()
    if not query:
        print("No query provided, exiting.")
        return

    # 7) Embed the query
    qvec = embedder.embed(query)

    # 8) Perform vector search in Qdrant
    hits = search(qvec, top_k=top_k)

    # 9) Display results
    print(f"\nTop {top_k} results for: '{query}'\n")
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        chunk_id = payload.get("chunk_id", hit.id)
        source   = payload.get("source", "<unknown>")
        strategy = payload.get("strategy", "<unknown>")
        score    = hit.score
        print(f"{rank:2d}. {chunk_id} (source={source}, strategy={strategy}) → score={score:.4f}")

if __name__ == "__main__":
    main()
