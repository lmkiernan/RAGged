import os
import json
import sys
import argparse
import logging
from supabase_client import SupabaseClient
from embedding_router import embed
from config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_chunks(user_id: str):
    """Process chunks from Supabase storage and generate embeddings."""
    try:
        # Initialize Supabase client
        supabase = SupabaseClient()
        
        # Load configuration
        config = load_config("config/default.yaml")
        
        # Get all chunk files for the user
        chunk_files = supabase.list_files(user_id, prefix=f"chunks/{user_id}/")
        if not chunk_files:
            logger.warning(f"No chunk files found for user {user_id}")
            return
            
        # Process each chunk file
        for file_info in chunk_files:
            file_path = file_info['name']
            if not file_path.endswith('.json'):
                continue
                
            logger.info(f"Processing chunks from: {file_path}")
            
            # Download the chunk file
            chunk_data = supabase.download_file(file_path, user_id)
            if not chunk_data:
                logger.warning(f"Failed to download chunk file: {file_path}")
                continue
                
            chunks = json.loads(chunk_data.decode('utf-8'))
            
            # Add user_id to each chunk
            for chunk in chunks:
                chunk['user_id'] = user_id
                
            # Generate embeddings
            for chunk in chunks:
                try:
                    embed(chunk, config)
                    logger.info(f"Successfully embedded chunk: {chunk['chunk_id']}")
                except Exception as e:
                    logger.error(f"Error embedding chunk {chunk['chunk_id']}: {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error processing chunks: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for chunks')
    parser.add_argument('--user-id', required=True, help='User ID for storage')
    args = parser.parse_args()
    
    try:
        process_chunks(args.user_id)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


