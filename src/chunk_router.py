#Implements the three chunking strategies (fixed-token, sliding-window, sentence-aware). 
#Takes ingested text and outputs a list of chunks (with start/end positions).

from src.chunking.sentence_aware import sentence_aware_chunk

def chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:
    if config["chunking"]["strategy"] == "fixed_token":
        return fixed_token_chunk(text, config)
    elif config["chunking"]["strategy"] == "sliding_window":
        return sliding_window_chunk(text, config)
    elif config["chunking"]["strategy"] == "sentence_aware":
        return sentence_aware_chunk(text, doc_id, config, model_provider_map)
    else:
        raise ValueError(f"Invalid chunking strategy: {config['chunking']['strategy']}")
    
def fixed_token_chunk(text: str, config: dict) -> list[dict]:
    return []

def sliding_window_chunk(text: str, config: dict) -> list[dict]:
    return []
