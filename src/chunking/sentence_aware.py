import spacy
from src.tokenizer import get_token_counts

nlp = spacy.load("en_core_web_sm")

def sentence_aware_chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:
    max_tokens = config["chunking"]["sentence_max_tokens"]

    # 1. Split into sentences once
    doc = nlp(text)
    sentence_objs = [
        (sent.text, sent.start_char, sent.end_char)
        for sent in doc.sents
    ]
    # 2. Batch‐tokenize sentences per model (huggingface / openai separately) if you have many
    #    Otherwise, just do one‐by‐one:
    sent_token_counts = {
        sent_text: max(get_token_counts(sent_text, model_provider_map).values())
        for sent_text, _, _ in sentence_objs
    }

    chunks = []
    buffer = []
    buffer_tokens = 0
    chunk_index = 0

    for sent_text, start_c, end_c in sentence_objs:
        sent_tokens = sent_token_counts[sent_text]

        if buffer and (buffer_tokens + sent_tokens > max_tokens):
            # finalize
            chunk_index += 1
            first_text, first_start, _ = buffer[0]
            last_text, _, last_end = buffer[-1]
            chunk_text = " ".join([b[0] for b in buffer])
            chunk_tokens = get_token_counts(chunk_text, model_provider_map)

            chunks.append({
                "chunk_id":   f"{doc_id}_chunk_{chunk_index}",
                "text":       chunk_text,
                "char_start": first_start,
                "char_end":   last_end,
                "source":     doc_id,
                "strategy":   "sentence_aware",
                "tokens":     chunk_tokens
            })

            buffer = [(sent_text, start_c, end_c)]
            buffer_tokens = sent_tokens
        else:
            buffer.append((sent_text, start_c, end_c))
            buffer_tokens += sent_tokens

    # finalize leftover
    if buffer:
        chunk_index += 1
        first_text, first_start, _ = buffer[0]
        last_text, _, last_end = buffer[-1]
        chunk_text = " ".join([b[0] for b in buffer])
        chunk_tokens = get_token_counts(chunk_text, model_provider_map)

        chunks.append({
            "chunk_id":   f"{doc_id}_chunk_{chunk_index}",
            "text":       chunk_text,
            "char_start": first_start,
            "char_end":   last_end,
            "source":     doc_id,
            "strategy":   "sentence_aware",
            "tokens":     chunk_tokens
        })

    return chunks