import os
import json
import yaml
import re
import string
from openai import OpenAI, ChatCompletion
from src.config import load_config

def load_api_keys():
    try:
        with open("APIKeys.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("APIKeys.json not found. Please ensure it exists in the root directory.")
    except json.JSONDecodeError:
        raise ValueError("APIKeys.json is not valid JSON.")

def generate_queries(doc_id: str, text: str, num_qs : int = 3) -> list[dict]:
    prompt = f"""
        Here is the text of a document named (ID: {doc_id}):
        {text}
        Please generate {num_qs} concise, factual question-answer pairs based on this document.  
        Respond in JSON as a list of objects with keys:\n  - question: string  \n  - answer: string (exact span from the document)\nEnsure the JSON is valid and nothing else is included.
        """
    
    api_keys = load_api_keys()
    api_key = api_keys.get("openai")
    if not api_key:
        raise ValueError("OpenAI API key not found in APIKeys.json")
        
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    print(f"\nDebug - Raw response from GPT:\n{content}\n")
    
    try:
        # First try to parse as a JSON object
        parsed = json.loads(content)
        print(f"Debug - Parsed JSON type: {type(parsed)}")
        print(f"Debug - Parsed JSON content: {parsed}")
        
        # Handle different response formats
        if isinstance(parsed, dict):
            # Try different possible keys
            for key in ['questions', 'qa_pairs', 'question-answer pairs']:
                if key in parsed:
                    qa_pairs = parsed[key]
                    break
            else:
                raise ValueError(f"Could not find QA pairs in response. Available keys: {list(parsed.keys())}")
        elif isinstance(parsed, list):
            qa_pairs = parsed
        else:
            raise ValueError(f"Unexpected response format: {type(parsed)}")
            
        # Validate the structure
        if not isinstance(qa_pairs, list):
            raise ValueError("Expected a list of question-answer pairs")
            
        for qa in qa_pairs:
            if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
                raise ValueError("Each QA pair must be a dict with 'question' and 'answer' keys")
                
        return qa_pairs
        
    except json.JSONDecodeError:
        # Try to extract JSON array from the response
        m = re.search(r"\[(.*)\]", content, re.DOTALL)
        if m:
            try:
                qa_pairs = json.loads(m.group(0))
                if not isinstance(qa_pairs, list):
                    raise ValueError("Expected a list of question-answer pairs")
                return qa_pairs
            except json.JSONDecodeError:
                raise ValueError("Could not parse response as JSON")
        else:
            raise ValueError("Could not find JSON array in response")

def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def map_answers_to_chunks(doc_id: str, qa_pairs: list[dict], chunks_path: str) -> list[dict]:
    mapped = []
    for fname in os.listdir(chunks_path):
        if not fname.startswith(doc_id + '_') or not fname.endswith('.json'):
            continue
        chunks = json.load(open(os.path.join(chunks_path, fname), 'r', encoding='utf-8'))
        for qa in qa_pairs:
            ans = normalize(qa['answer'])
            for chunk in chunks:
                if ans and ans in normalize(chunk['text']):
                    mapped.append({
                        'question': qa['question'],
                        'gold_chunk_id': chunk['chunk_id'],
                        'strategy': chunk['strategy']
                    })
                    break
    return mapped

def main():
    # Load config for data paths
    cfg = load_config('config/default.yaml')
    data_folder = cfg['ingestion']['input_folder']
    chunks_folder = 'chunks'
    output_file = 'gold_queries.yaml'

    gold = {}
    # Process each ingested document
    for fname in os.listdir(data_folder):
        doc_id, ext = os.path.splitext(fname)
        if ext.lower() not in ['.pdf', '.md', '.html']:
            continue
        # Load text
        doc_json = json.load(open(os.path.join('ingested', doc_id + '.json'), 'r', encoding='utf-8'))
        text = doc_json['text']
        # Generate QA pairs
        qa_pairs = generate_queries(doc_id, text, cfg['evaluation']['num_questions_per_doc'])
        # Map to chunks
        mapped = map_answers_to_chunks(doc_id, qa_pairs, chunks_folder)
        gold[doc_id + '.json'] = mapped

    # Write to YAML
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(gold, f, sort_keys=False)
    print(f"Gold queries written to {output_file}")


if __name__ == '__main__':
    main()