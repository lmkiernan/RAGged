import os
import sys
import json
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.chunk_router import chunk
from src.config import load_config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chunk documents using specified strategy')
    parser.add_argument('--strategy', required=True, choices=['sentence_aware', 'fixed_token', 'sliding_window'],
                      help='Chunking strategy to use')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config('config/default.yaml')
    
    # Set up directories
    ingested_dir = os.path.join(os.path.dirname(__file__), '..', 'ingested')
    chunks_dir = os.path.join(os.path.dirname(__file__), '..', 'chunks')
    
    # Ensure chunks directory exists
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir, exist_ok=True)
    
    # Model provider mapping
    model_provider_map = { 
        "BAAI/bge-large-en": "huggingface",
        "text-embedding-3-small": "openai"
    }

    print(f"\nChunking documents using {args.strategy} strategy...")
    
    # Process each ingested document
    for filename in os.listdir(ingested_dir):
        if not filename.endswith('.json'):
            continue
            
        doc_id = os.path.splitext(filename)[0]
        print(f"\nProcessing: {doc_id}")
        
        try:
            # Load the ingested document
            with open(os.path.join(ingested_dir, filename), 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Generate chunks
            chunks = chunk(doc_data['text'], doc_id, cfg, model_provider_map, args.strategy)
            
            if not chunks:
                print(f"Warning: No chunks generated for {doc_id}")
                continue
                
            # Save chunks
            output_path = os.path.join(chunks_dir, f"{doc_id}_{args.strategy}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(chunks)} chunks to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {doc_id}: {str(e)}")

if __name__ == "__main__":
    main()
