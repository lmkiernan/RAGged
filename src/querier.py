import os
import json
import yaml
import re
import string
from openai import OpenAI, ChatCompletion
from .config import load_config
from .supabase_client import SupabaseClient
import logging
import argparse
import sys
from typing import List, Dict, Any
import traceback
import time
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('querier.log')
    ]
)
logger = logging.getLogger(__name__)

def get_api_key():
    """Get API key from environment variable."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def generate_queries(text: str, num_qs : int = 5) -> list[dict]:
    prompt = f"""
        Here is the text of a document:
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
    
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    logger.info(f"Generating {num_qs} QA pairs for document {api_key}")
    
    logger.info("Making API call to GPT-4...")
        
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    
    json_content = json.loads(content)

    return json_content
            

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
        
        # Create the storage path with the correct prefix
        storage_path = f"qa_pairs/{user_id}/{doc_id}_qa.json"
        logger.info(f"Saving QA pairs to path: {storage_path}")
        
        # Save to Supabase
        qa_json = json.dumps(qa_pairs, ensure_ascii=False)
        result = supabase.supabase.storage.from_('documents').upload(
            storage_path,
            qa_json.encode('utf-8'),
            {'content-type': 'application/json'}
        )
        
        if not result:
            raise Exception("Failed to upload QA pairs to Supabase")
            
        logger.info(f"Successfully saved {len(qa_pairs)} QA pairs to Supabase: {storage_path}")
        return storage_path
        
    except Exception as e:
        logger.error(f"Error saving QA pairs: {str(e)}")
        raise

def process_document(doc_id: str, doc_data: dict, user_id: str, num_questions: int = 5) -> str:
    """Process a document to generate and save QA pairs."""
    try:
        logger.info(f"Starting QA pair generation for document {doc_id}")
        
        # Generate QA pairs
        logger.info(f"Generating {num_questions} QA pairs for document {doc_id}")
        qa_pairs = generate_queries(doc_id, doc_data['text'], num_questions)
        logger.info(f"Generated {len(qa_pairs)} QA pairs")
        
        # Map answers to chunks
        logger.info(f"Mapping answers to chunks for document {doc_id}")
        mapped_qa = map_answers_to_chunks(doc_id, qa_pairs, user_id)
        logger.info(f"Mapped {len(mapped_qa)} QA pairs to chunks")
        
        # Save QA pairs
        logger.info(f"Saving QA pairs for document {doc_id}")
        storage_path = save_qa_pairs(mapped_qa, doc_id, user_id)
        logger.info(f"Successfully saved QA pairs to {storage_path}")
        
        return storage_path
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate QA pairs for documents')
    parser.add_argument('--user-id', required=True, help='User ID for storage')
    parser.add_argument('--num-questions', type=int, default=5, help='Number of questions per document')
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting QA pair generation for user {args.user_id}")
        
        # Initialize Supabase client
        supabase = SupabaseClient()
        
        # Get all processed files for the user
        logger.info(f"Listing processed files for user {args.user_id}")
        processed_files = supabase.list_files(args.user_id, prefix="processed/")
        if not processed_files:
            logger.warning(f"No processed files found for user {args.user_id}")
            return
            
        logger.info(f"Found {len(processed_files)} processed files")
        
        # Process each file
        for file_info in processed_files:
            try:
                file_path = file_info['name']
                if not file_path.endswith('.json'):
                    logger.debug(f"Skipping non-JSON file: {file_path}")
                    continue
                    
                # Get document ID from filename
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                logger.info(f"Processing file: {file_path} (doc_id: {doc_id})")
                
                # Download the processed file
                logger.info(f"Downloading file: {file_path}")
                file_data = supabase.download_file(file_path, args.user_id, prefix="processed/")
                if not file_data:
                    raise ValueError(f"Failed to download file: {file_path}")
                    
                # Parse the JSON
                doc_data = json.loads(file_data.decode('utf-8'))
                logger.info(f"Successfully loaded document data for {doc_id}")
                
                # Process the document
                qa_path = process_document(doc_id, doc_data, args.user_id, args.num_questions)
                logger.info(f"Successfully processed {file_path} -> {qa_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()