from src.tokenizer import get_token_counts
import spacy

def sentence_aware_chunk(text: str, doc_id: str, config: dict, model_provider_map: dict) -> list[dict]:

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentence_list = []
    i = 1
    doc_id = doc_id.removesuffix(".json")
    for sent in doc.sents:

        sentence_list.append(
            {
            "chunk_id": doc_id + "_" + str(i),
            "text": sent.text,
             "char_start": sent.start_char,
             "char_end": sent.end_char,
             "source": doc_id,
             "strategy": "sentence_aware",
             "tokens": get_token_counts(sent.text, model_provider_map)
            }
            )
        i += 1
    return sentence_list