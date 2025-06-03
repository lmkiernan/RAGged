#Implements the three chunking strategies (fixed-token, sliding-window, sentence-aware). 
#Takes ingested text and outputs a list of chunks (with start/end positions).

from src.chunking.sentence_aware import sentence_aware_chunk
from src.chunking.fixed_token import fixed_token_chunk
from src.chunking.sliding_window import sliding_window_chunk

def chunk(text: str, doc_id: str, config: dict, model_provider_map: dict, strat: str) -> list[dict]:
    if strat == "fixed_token":
        return fixed_token_chunk(text, doc_id, config, model_provider_map)
    elif strat == "sliding_window":
        return sliding_window_chunk(text, doc_id, config, model_provider_map)
    elif strat == "sentence_aware":
        return sentence_aware_chunk(text, doc_id, config, model_provider_map)
    else:
        raise ValueError(f"Invalid chunking strategy: {strat}")
    


