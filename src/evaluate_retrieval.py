import os
import json
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import load_config
from src.Embedding import OpenAIEmbedder, HFEmbedder
from src.vectorStore import search

def load_api_keys():
    try:
        with open("APIKeys.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("APIKeys.json not found. Please ensure it exists in the root directory.")
    except json.JSONDecodeError:
        raise ValueError("APIKeys.json is not valid JSON.")

def evaluate_retrieval():
    # Load configuration
    cfg = load_config("config/default.yaml")
    api_keys = load_api_keys()
    
    # Initialize embedder (using first provider in config)
    emb_config = cfg["embedding"][0]
    provider = emb_config["provider"].lower()
    
    if provider == "openai":
        model_name = cfg["openai"][0]["model"]
        embedder = OpenAIEmbedder(model_name, api_keys["openai"])
    elif provider in ("huggingface", "hf"):
        model_name = cfg["huggingface"][0]["model"]
        embedder = HFEmbedder(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Get retrieval parameters
    top_k = cfg.get("objectives", {}).get("retrieval_top_k", 5)
    
    # Process each golden questions file
    golden_qs_dir = os.path.join(os.path.dirname(__file__), '..', 'golden_qs')
    results = {
        "total_questions": 0,
        "found_in_top_k": 0,
        "rank_distribution": {},  # How often golden chunk appears at each rank
        "by_strategy": {}  # Results broken down by chunking strategy
    }
    
    for filename in os.listdir(golden_qs_dir):
        if not filename.endswith('_golden.json'):
            continue
            
        doc_id = filename.replace('_golden.json', '')
        print(f"\nEvaluating questions for: {doc_id}")
        
        # Load golden questions
        with open(os.path.join(golden_qs_dir, filename), 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        results["total_questions"] += len(questions)
        
        # Process each question
        for q in questions:
            strategy = q["strategy"]
            if strategy not in results["by_strategy"]:
                results["by_strategy"][strategy] = {
                    "total": 0,
                    "found_in_top_k": 0,
                    "rank_distribution": {}
                }
            
            results["by_strategy"][strategy]["total"] += 1
            
            # Get query embedding and search
            query_vector = embedder.embed(q["question"])
            hits = search(query_vector, top_k=top_k)
            
            # Find where the golden chunk appears
            found = False
            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}
                chunk_id = payload.get("chunk_id", hit.id)
                
                if chunk_id == q["gold_chunk_id"]:
                    found = True
                    results["found_in_top_k"] += 1
                    results["by_strategy"][strategy]["found_in_top_k"] += 1
                    
                    # Update rank distribution
                    results["rank_distribution"][rank] = results["rank_distribution"].get(rank, 0) + 1
                    results["by_strategy"][strategy]["rank_distribution"][rank] = \
                        results["by_strategy"][strategy]["rank_distribution"].get(rank, 0) + 1
                    
                    print(f"✓ Question: {q['question']}")
                    print(f"  Found golden chunk at rank {rank}")
                    break
            
            if not found:
                print(f"✗ Question: {q['question']}")
                print(f"  Golden chunk not found in top {top_k} results")
    
    # Calculate and display metrics
    print("\n=== Retrieval Evaluation Results ===")
    print(f"Total questions evaluated: {results['total_questions']}")
    print(f"Found in top {top_k}: {results['found_in_top_k']} ({results['found_in_top_k']/results['total_questions']*100:.1f}%)")
    
    print("\nRank Distribution:")
    for rank in range(1, top_k + 1):
        count = results["rank_distribution"].get(rank, 0)
        print(f"Rank {rank}: {count} ({count/results['total_questions']*100:.1f}%)")
    
    print("\nResults by Chunking Strategy:")
    for strategy, stats in results["by_strategy"].items():
        print(f"\n{strategy}:")
        print(f"  Total questions: {stats['total']}")
        print(f"  Found in top {top_k}: {stats['found_in_top_k']} ({stats['found_in_top_k']/stats['total']*100:.1f}%)")
        print("  Rank Distribution:")
        for rank in range(1, top_k + 1):
            count = stats["rank_distribution"].get(rank, 0)
            print(f"    Rank {rank}: {count} ({count/stats['total']*100:.1f}%)")

if __name__ == "__main__":
    evaluate_retrieval() 