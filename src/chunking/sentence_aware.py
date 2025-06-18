import spacy
import os
from ..tokenizer import count_tokens
nlp = spacy.load("en_core_web_sm")

def sentence_aware_chunk(text: str, doc_id: str, config: dict, model_name: str, provider: str) -> list[dict]:
    max_tokens = config["sentence_max_tokens"]

    # Split into sentences
    doc = nlp(text)
    sentence_objs = [
        (sent.text, sent.start_char, sent.end_char)
        for sent in doc.sents
    ]

    chunks = []
    buffer = []
    buffer_tokens = 0
    chunk_index = 0

    for sent_text, start_c, end_c in sentence_objs:
        # Estimate tokens (rough count)
        sent_tokens = len(sent_text.split()) * 1.3  # Rough estimate
        doc = os.path.splitext(doc_id)[0]
        
        if buffer and (buffer_tokens + sent_tokens > max_tokens):
            # finalize current chunk
            chunk_index += 1
            first_text, first_start, _ = buffer[0]
            last_text, _, last_end = buffer[-1]
            chunk_text = " ".join([b[0] for b in buffer])

            chunks.append({
                "chunk_id": f"{doc}_sa_{chunk_index}",
                "text": chunk_text,
                "char_start": first_start,
                "char_end": last_end,
                "strategy": "sentence_aware",
                "source": doc_id,
                "model": model_name,
                "provider": provider,
                "token_count": len(chunk_text.split())
            })

            buffer = [(sent_text, start_c, end_c)]
            buffer_tokens = sent_tokens
        else:
            buffer.append((sent_text, start_c, end_c))
            buffer_tokens += sent_tokens

    # Handle remaining sentences
    if buffer:
        chunk_index += 1
        first_text, first_start, _ = buffer[0]
        last_text, _, last_end = buffer[-1]
        chunk_text = " ".join([b[0] for b in buffer])

        chunks.append({
            "chunk_id": f"{doc}_sa_{chunk_index}",
            "text": chunk_text,
            "char_start": first_start,
            "char_end": last_end,
            "strategy": "sentence_aware",
            "source": doc_id,
            "model": model_name,
            "provider": provider,
            "token_count": count_tokens(chunk_text)
        })

    return chunks