import os
import json
import yaml
import re
import string
from openai import OpenAI, ChatCompletion
from src.config import load_config
from supabase_client import SupabaseClient
import logging
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        IMPORTANT: You must respond with a JSON array containing exactly {num_qs} objects.
        Each object must have exactly these two keys:
        - "question": string
        - "answer": string (exact span from the document)
        
        Example format:
        [
            {{"question": "What is X?", "answer": "X is..."}},
            {{"question": "How does Y work?", "answer": "Y works by..."}},
            {{"question": "When did Z happen?", "answer": "Z happened in..."}}
        ]
        
        Ensure the response is a valid JSON array and nothing else is included.
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
    logger.debug(f"Raw response from GPT:\n{content}")
    
    try:
        # First try to parse as a JSON object
        parsed = json.loads(content)
        logger.debug(f"Parsed JSON type: {type(parsed)}")
        logger.debug(f"Parsed JSON content: {parsed}")
        
        # Handle different response formats
        if isinstance(parsed, dict):
            # If it's a single QA pair, wrap it in a list
            if 'question' in parsed and 'answer' in parsed:
                return [parsed]
            
            # Look for any key that contains a list of QA pairs
            for value in parsed.values():
                if isinstance(value, list) and len(value) > 0:
                    # Check if the first item has the right structure
                    if isinstance(value[0], dict) and 'question' in value[0] and 'answer' in value[0]:
                        return value
            
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

def map_answers_to_chunks(doc_id: str, qa_pairs: list[dict], user_id: str) -> list[dict]:
    """Map answers to chunks stored in Supabase."""
    try:
        supabase = SupabaseClient()
        mapped = []
        
        # Get all chunk files for this document
        chunk_files = supabase.list_files(user_id, prefix=f"chunks/{user_id}/")
        if not chunk_files:
            logger.warning(f"No chunk files found for user {user_id}")
            return mapped
            
        for file_info in chunk_files:
            file_path = file_info['name']
            if not file_path.endswith('.json'):
                continue
                
            # Download the chunk file
            chunk_data = supabase.download_file(file_path, user_id)
            if not chunk_data:
                logger.warning(f"Failed to download chunk file: {file_path}")
                continue
                
            chunks = json.loads(chunk_data.decode('utf-8'))
            
            # Map answers to chunks
            for qa in qa_pairs:
                ans = normalize(qa['answer'])
                for chunk in chunks:
                    if ans and ans in normalize(chunk['text']):
                        mapped.append({
                            'question': qa['question'],
                            'gold_chunk_id': chunk['chunk_id'],
                            'strategy': chunk['strategy'],
                            'source': chunk['source']
                        })
                        break
                        
        return mapped
        
    except Exception as e:
        logger.error(f"Error mapping answers to chunks: {str(e)}")
        raise

def save_qa_pairs(qa_pairs: list[dict], doc_id: str, user_id: str) -> str:
    """Save QA pairs to Supabase storage."""
    try:
        supabase = SupabaseClient()
        
        # Create the storage path
        storage_path = f"qa_pairs/{user_id}/{doc_id}_qa.json"
        
        # Save to Supabase
        qa_json = json.dumps(qa_pairs, ensure_ascii=False)
        result = supabase.supabase.storage.from_('documents').upload(
            storage_path,
            qa_json.encode('utf-8'),
            {'content-type': 'application/json'}
        )
        
        logger.info(f"Saved QA pairs to Supabase: {storage_path}")
        return storage_path
        
    except Exception as e:
        logger.error(f"Error saving QA pairs: {str(e)}")
        raise

def process_document(doc_id: str, doc_data: dict, user_id: str, num_questions: int = 3) -> str:
    """Process a document to generate and save QA pairs."""
    try:
        # Generate QA pairs
        qa_pairs = generate_queries(doc_id, doc_data['text'], num_questions)
        
        # Map answers to chunks
        mapped_qa = map_answers_to_chunks(doc_id, qa_pairs, user_id)
        
        # Save QA pairs
        storage_path = save_qa_pairs(mapped_qa, doc_id, user_id)
        
        return storage_path
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate QA pairs for documents')
    parser.add_argument('--user-id', required=True, help='User ID for storage')
    parser.add_argument('--num-questions', type=int, default=3, help='Number of questions per document')
    args = parser.parse_args()
    
    try:
        # Initialize Supabase client
        supabase = SupabaseClient()
        
        # Get all processed files for the user
        processed_files = supabase.list_files(args.user_id, prefix="processed/")
        if not processed_files:
            logger.warning(f"No processed files found for user {args.user_id}")
            return
            
        # Process each file
        for file_info in processed_files:
            try:
                file_path = file_info['name']
                if not file_path.endswith('.json'):
                    continue
                    
                # Get document ID from filename
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                
                # Download the processed file
                file_data = supabase.download_file(file_path, args.user_id)
                if not file_data:
                    raise ValueError(f"Failed to download file: {file_path}")
                    
                # Parse the JSON
                doc_data = json.loads(file_data.decode('utf-8'))
                
                # Process the document
                qa_path = process_document(doc_id, doc_data, args.user_id, args.num_questions)
                logger.info(f"Successfully processed {file_path} -> {qa_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()