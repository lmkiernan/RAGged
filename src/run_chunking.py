import os
import sys
import json
import argparse
from typing import List, Dict, Any
import logging
from supabase_client import SupabaseClient
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chunking.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_config(config: dict) -> None:
    """Validate the configuration parameters."""
    required_fields = {
        "fixed_chunk_size": int,
        "overlap": int,
        "sentence_max_tokens": int
    }
    
    for field, field_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
        if not isinstance(config[field], field_type):
            raise ValueError(f"Invalid type for {field}. Expected {field_type}, got {type(config[field])}")
    
    if config["overlap"] >= config["fixed_chunk_size"]:
        raise ValueError("Overlap must be smaller than chunk size")

def validate_document(doc_data: Dict[str, Any]) -> None:
    """Validate the document data structure."""
    required_fields = ["text", "source"]
    for field in required_fields:
        if field not in doc_data:
            raise ValueError(f"Document missing required field: {field}")
    
    if not isinstance(doc_data["text"], str):
        raise ValueError("Document text must be a string")
    if not doc_data["text"].strip():
        raise ValueError("Document text cannot be empty")

def chunk_text(text: str, strategy: str, model_name: str, provider: str, config: dict) -> List[Dict[str, Any]]:
    """Chunk text based on the specified strategy."""
    try:
        if strategy == "fixed_token":
            from src.chunking.fixed_token import fixed_token_chunk
            return fixed_token_chunk(text, "temp", config, {}, "temp", model_name, provider)
        elif strategy == "sliding_window":
            from src.chunking.sliding_window import sliding_window_chunk
            return sliding_window_chunk(text, "temp", config, {}, "temp", model_name, provider)
        elif strategy == "sentence_aware":
            from src.chunking.sentence_aware import sentence_aware_chunk
            return sentence_aware_chunk(text, "temp", config, {}, "temp", model_name, provider)
        else:
            raise ValueError(f"Invalid chunking strategy: {strategy}")
    except ImportError as e:
        logger.error(f"Failed to import chunking module: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during chunking: {str(e)}")
        raise

def process_document(doc_data: Dict[str, Any], strategy: str, model_name: str, provider: str, user_id: str, config: dict) -> str:
    """Process a document and save its chunks to Supabase."""
    try:
        # Validate inputs
        validate_document(doc_data)
        validate_config(config)
        
        # Initialize Supabase client
        supabase = SupabaseClient()
        
        # Get the original filename
        original_path = doc_data.get('source', '')
        if not original_path:
            raise ValueError("Document missing source path")
            
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        chunks_filename = f"{base_name}_chunks.json"
        
        # Generate chunks
        logger.info(f"Generating chunks for {base_name} using {strategy} strategy with {model_name} model")
        chunks = chunk_text(doc_data['text'], strategy, model_name, provider, config)
        
        if not chunks:
            raise ValueError("No chunks were generated")
            
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = f"{base_name}_{strategy}_{i}"
            chunk['source'] = original_path
            chunk['strategy'] = strategy
            chunk['user_id'] = user_id
            chunk['model'] = model_name
            chunk['provider'] = provider
        
        # Save chunks to Supabase
        storage_path = f"chunks/{user_id}/{strategy}/{model_name}/{chunks_filename}"
        chunks_json = json.dumps(chunks, ensure_ascii=False)
        
        try:
            result = supabase.supabase.storage.from_('documents').upload(
                storage_path,
                chunks_json.encode('utf-8'),
                {'content-type': 'application/json'}
            )
            logger.info(f"Successfully saved chunks to Supabase: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to upload chunks to Supabase: {str(e)}")
            raise
        
        return storage_path
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Chunk documents using specified strategy')
    parser.add_argument('--strategy', required=True, help='Chunking strategy to use')
    parser.add_argument('--model', required=True, help='Model name to use')
    parser.add_argument('--provider', required=True, help='Model provider (huggingface/openai)')
    parser.add_argument('--user-id', required=True, help='User ID for storage')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    try:
        # Load and validate config
        with open(args.config, 'r') as f:
            config = json.load(f)
        validate_config(config)
        
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
                    logger.info(f"Skipping non-JSON file: {file_path}")
                    continue
                    
                logger.info(f"Processing file: {file_path}")
                
                # Download the processed file
                file_data = supabase.download_file(file_path, args.user_id)
                if not file_data:
                    raise ValueError(f"Failed to download file: {file_path}")
                    
                # Parse the JSON
                try:
                    doc_data = json.loads(file_data.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
                    continue
                
                # Process the document
                chunks_path = process_document(doc_data, args.strategy, args.model, args.provider, args.user_id, config)
                logger.info(f"Successfully processed {file_path} -> {chunks_path}")
                
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
