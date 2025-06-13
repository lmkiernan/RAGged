import os
import json
from src.config import load_config
from src.Embedding import OpenAIEmbedder, HFEmbedder
from src.vectorStore import search

def main():
    # 1) Load configuration
    cfg = load_config("config/default.yaml")

    # 2) Determine which embedder to use (pick the first in the list)
    emb_configs = cfg.get("embedding", [])
    if not emb_configs:
        raise ValueError("No embedding configuration found in config/default.yaml")

    primary = emb_configs[0]
    provider = primary.get("provider").lower()

    # 3) Instantiate the appropriate Embedder
    if provider == "openai":
        # Expecting OPENAI_API_KEY in env
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set")
        model_name = cfg.get("openai")[0].get("model")
        embedder = OpenAIEmbedder(model_name, api_key)
    elif provider in ("huggingface", "hf"):  
        model_name = cfg.get("huggingface")[0].get("model")
        embedder = HFEmbedder(model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    # 4) Load retrieval parameters
    top_k = cfg.get("objectives", {}).get("retrieval_top_k", 5)

    # 5) Prompt user for a query
    query = input("\nEnter a natural-language query: ").strip()
    if not query:
        print("No query provided, exiting.")
        return

    # 6) Embed the query
    qvec = embedder.embed(query)

    # 7) Perform vector search in Qdrant
    hits = search(qvec, top_k=top_k)

    # 8) Display results
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
