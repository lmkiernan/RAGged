import os
import sys
import json

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ingest import ingest_file
from src.querier import generate_queries

def main():
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    ingested_dir = os.path.join(os.path.dirname(__file__), '..', 'ingested')
    og_qa_dir = os.path.join(os.path.dirname(__file__), '..', 'querying', 'og_qa')
    
    # Ensure directories exist
    for directory in [data_dir, ingested_dir, og_qa_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    # First, ingest all files
    print("Step 1: Ingesting files...")
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
    
    # Now generate QA pairs for each ingested document
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

if __name__ == "__main__":
    main()
