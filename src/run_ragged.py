import os
import sys
import json
import argparse
import subprocess
import traceback
import requests
import shutil

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from embedding_router import get_embedder
from vectorStore import search
from config import load_config
from ingest import ingest_file
from querier import generate_queries, map_answers_to_chunks

def clear_directories():
    """Clear all files from the working directories."""
    directories = [
        os.path.join(project_root, 'ingested'),
        os.path.join(project_root, 'querying', 'og_qa'),
        os.path.join(project_root, 'chunks'),
        os.path.join(project_root, 'golden_qs')
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"Clearing directory: {directory}")
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        else:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

def clear_qdrant():
    """Clear all points from the Qdrant collection."""
    try:
        response = requests.post(
            'http://localhost:6333/collections/autoembed_chunks/points/delete',
            json={"filter": {}},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            print("Successfully cleared Qdrant collection")
        else:
            print(f"Warning: Failed to clear Qdrant collection. Status code: {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant: {str(e)}")

def query_golden_questions(cfg):
    """Query Qdrant with each question from golden_qs files."""
    print("\nStep 5: Querying with golden questions...")
    
    # Get embedder for querying
    embedder = get_embedder("openai", cfg["openai"][0]["model"])
    top_k = cfg.get("objectives", {}).get("retrieval_top_k", 5)
    
    # Process each golden questions file
    golden_qs_dir = os.path.join(os.path.dirname(__file__), '..', 'golden_qs')
    for filename in os.listdir(golden_qs_dir):
        if not filename.endswith('_golden.json'):
            continue
            
        doc_id = filename.replace('_golden.json', '')
        print(f"\nProcessing questions for: {doc_id}")
        
        try:
            # Load the golden questions
            with open(os.path.join(golden_qs_dir, filename), 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # Query each question
            for q in questions:
                print(f"\nQuestion: {q['question']}")
                print(f"Expected chunk ID: {q['gold_chunk_id']}")
                
                # Get query embedding and search
                query_vector = embedder.embed(q['question'])
                hits = search(query_vector, top_k=top_k)
                
                # Display results
                print(f"\nTop {top_k} results:")
                for rank, hit in enumerate(hits, start=1):
                    payload = hit.payload or {}
                    chunk_id = payload.get("chunk_id", hit.id)
                    source = payload.get("source", "<unknown>")
                    strategy = payload.get("strategy", "<unknown>")
                    score = hit.score
                    print(f"{rank:2d}. {chunk_id} (source={source}, strategy={strategy}) → score={score:.4f}")
                    print(f"   Text: {payload.get('text', '')[:200]}...")
                    
        except Exception as e:
            print(f"Error processing questions for {doc_id}: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())

def main():
    # Clear directories and Qdrant at the start
    print("Step 0: Clearing directories and Qdrant collection...")
    clear_directories()
    clear_qdrant()
    
    # Load configuration
    cfg = load_config('config/default.yaml')
    
    # Get the directory paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    ingested_dir = os.path.join(os.path.dirname(__file__), '..', 'ingested')
    og_qa_dir = os.path.join(os.path.dirname(__file__), '..', 'querying', 'og_qa')
    chunks_dir = os.path.join(os.path.dirname(__file__), '..', 'chunks')
    golden_qs_dir = os.path.join(os.path.dirname(__file__), '..', 'golden_qs')
    
    # Ensure directories exist
    for directory in [data_dir, ingested_dir, og_qa_dir, chunks_dir, golden_qs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    # First, ingest all files
    print("\nStep 1: Ingesting files...")
    supported_extensions = {'.pdf', '.md', '.html'}
    files_to_process = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]
    
    if not files_to_process:
        print("No supported files found in data directory.")
        return
    
    # Process each file
    print(f"Found {len(files_to_process)} files to process:")
    for filename in files_to_process:
        file_path = os.path.join(data_dir, filename)
        print(f"\nProcessing: {filename}")
        try:
            ingested_path = ingest_file(file_path)
            print(f"Successfully ingested to: {ingested_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
    
    # Generate QA pairs for each ingested document
    print("\nStep 2: Generating QA pairs...")
    for filename in os.listdir(ingested_dir):
        if not filename.endswith('.json'):
            continue
            
        doc_id = os.path.splitext(filename)[0]
        print(f"\nGenerating QA pairs for: {doc_id}")
        
        try:
            # Load the ingested document
            with open(os.path.join(ingested_dir, filename), 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Generate QA pairs
            qa_pairs = generate_queries(doc_id, doc_data['text'], num_qs=4)
            
            # Save QA pairs
            output_path = os.path.join(og_qa_dir, f"{doc_id}_qa.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            print(f"Saved QA pairs to: {output_path}")
            
        except Exception as e:
            print(f"Error generating QA pairs for {doc_id}: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())

    # Run chunking with the first strategy from config
    print("\nStep 3: Running chunking with first strategy...")
    first_strategy = cfg["strats"][0]
    print(f"Using strategy: {first_strategy}")
    print(f"Chunking config: {json.dumps(cfg, indent=2)}")
    
    try:
        # Run run_chunking.py as a subprocess
        chunking_script = os.path.join(os.path.dirname(__file__), 'run_chunking.py')
        result = subprocess.run(['python3', chunking_script, '--strategy', first_strategy], 
                              capture_output=True, text=True)
        
        # Print the output
        if result.stdout:
            print("Chunking stdout:")
            print(result.stdout)
        if result.stderr:
            print("Chunking stderr:")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"\nSuccessfully completed chunking with {first_strategy} strategy")
        else:
            print(f"\nError during chunking with {first_strategy} strategy")
            return
            
    except Exception as e:
        print(f"Error running chunking script: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return

    # Map answers to chunks and save to golden_qs
    print("\nStep 4: Mapping answers to chunks...")
    for filename in os.listdir(og_qa_dir):
        if not filename.endswith('_qa.json'):
            continue
            
        doc_id = filename.replace('_qa.json', '')
        print(f"\nMapping answers for: {doc_id}")
        
        try:
            # Load the QA pairs
            with open(os.path.join(og_qa_dir, filename), 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            # Map answers to chunks
            mapped_answers = map_answers_to_chunks(doc_id, qa_pairs, chunks_dir)
            
            # Save mapped answers
            output_path = os.path.join(golden_qs_dir, f"{doc_id}_golden.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mapped_answers, f, indent=2, ensure_ascii=False)
            print(f"Saved mapped answers to: {output_path}")
            
        except Exception as e:
            print(f"Error mapping answers for {doc_id}: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            
    # Run embeddings
    print("\nStep 5: Running embeddings...")
    try:
        embeddings_script = os.path.join(os.path.dirname(__file__), 'run_embeddings.py')
        result = subprocess.run(['python3', embeddings_script], 
                              capture_output=True, text=True)
        
        if result.stdout:
            print("Embeddings stdout:")
            print(result.stdout)
        if result.stderr:
            print("Embeddings stderr:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\nSuccessfully completed embeddings")
        else:
            print("\nError during embeddings")
            return
            
    except Exception as e:
        print(f"Error running embeddings script: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return
            
    # Query with golden questions
    query_golden_questions(cfg)

if __name__ == "__main__":
    main()
