import os
import sys
import json
import argparse
from typing import List, Dict, Any
import logging
from supabase_client import SupabaseClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, strategy: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text based on the specified strategy."""
    chunks = []
    
    if strategy == "fixed_token":
        # Simple fixed-size chunking
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk,
                "start": i,
                "end": min(i + chunk_size, len(words))
            })
            
    elif strategy == "sliding_window":
        # Sliding window chunking
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk,
                "start": i,
                "end": min(i + chunk_size, len(words))
            })
            
    elif strategy == "sentence_aware":
        # Split into sentences first
        sentences = text.split('. ')
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": " ".join(current_chunk),
                    "start": len(" ".join(chunks)) if chunks else 0,
                    "end": len(" ".join(chunks)) + len(" ".join(current_chunk))
                })
                # Start new chunk with overlap
                overlap_words = " ".join(current_chunk[-overlap:]).split()
                current_chunk = overlap_words
                current_size = len(overlap_words)
            
            current_chunk.append(sentence)
            current_size += sentence_size
            
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "start": len(" ".join(chunks)) if chunks else 0,
                "end": len(" ".join(chunks)) + len(" ".join(current_chunk))
            })
            
    return chunks

def process_document(doc_data: Dict[str, Any], strategy: str, user_id: str) -> str:
    """Process a document and save its chunks to Supabase."""
    try:
        # Initialize Supabase client
        supabase = SupabaseClient()
        
        # Get the original filename
        original_path = doc_data.get('source', '')
        if not original_path:
            raise ValueError("Document missing source path")
            
        base_name = os.path.basename(original_path)
        chunks_filename = f"{base_name}_chunks.json"
        
        # Generate chunks
        chunks = chunk_text(doc_data['text'], strategy)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = f"{base_name}_{strategy}_{i}"
            chunk['source'] = original_path
            chunk['strategy'] = strategy
            chunk['user_id'] = user_id
        
        # Save chunks to Supabase
        storage_path = f"chunks/{user_id}/{strategy}/{chunks_filename}"
        chunks_json = json.dumps(chunks, ensure_ascii=False)
        
        result = supabase.supabase.storage.from_('documents').upload(
            storage_path,
            chunks_json.encode('utf-8'),
            {'content-type': 'application/json'}
        )
        
        logger.info(f"Saved chunks to Supabase: {storage_path}")
        return storage_path
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Chunk documents using specified strategy')
    parser.add_argument('--strategy', required=True, help='Chunking strategy to use')
    parser.add_argument('--user-id', required=True, help='User ID for storage')
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
                    
                # Download the processed file
                file_data = supabase.download_file(file_path, args.user_id)
                if not file_data:
                    raise ValueError(f"Failed to download file: {file_path}")
                    
                # Parse the JSON
                doc_data = json.loads(file_data.decode('utf-8'))
                
                # Process the document
                chunks_path = process_document(doc_data, args.strategy, args.user_id)
                logger.info(f"Successfully processed {file_path} -> {chunks_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
