import os
import json
import sys
from datetime import datetime

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
    
    # Initialize metrics collection
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_provider": provider,
            "embedding_model": model_name,
            "chunking_strategy": cfg["strats"][0],
            "top_k": top_k
        },
        "overall": {
            "total_questions": 0,
            "found_in_top_k": 0,
            "total_latency_ms": 0,
            "total_cost": 0,
            "rank_distribution": {},
            "recall_at_k": 0.0,
            "mean_reciprocal_rank": 0.0
        },
        "by_strategy": {}
    }
    
    # Process each golden questions file
    golden_qs_dir = os.path.join(os.path.dirname(__file__), '..', 'golden_qs')
    for filename in os.listdir(golden_qs_dir):
        if not filename.endswith('_golden.json'):
            continue
            
        doc_id = filename.replace('_golden.json', '')
        print(f"\nEvaluating questions for: {doc_id}")
        
        # Load golden questions
        with open(os.path.join(golden_qs_dir, filename), 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        metrics["overall"]["total_questions"] += len(questions)
        
        # Process each question
        for q in questions:
            strategy = q["strategy"]
            if strategy not in metrics["by_strategy"]:
                metrics["by_strategy"][strategy] = {
                    "total": 0,
                    "found_in_top_k": 0,
                    "total_latency_ms": 0,
                    "total_cost": 0,
                    "rank_distribution": {},
                    "recall_at_k": 0.0,
                    "mean_reciprocal_rank": 0.0
                }
            
            metrics["by_strategy"][strategy]["total"] += 1
            
            # Get query embedding and search
            query_vector = embedder.embed(q["question"])
            hits = search(query_vector, top_k=top_k)
            
            # Calculate cost (if using OpenAI)
            cost = 0
            if provider == "openai":
                # Approximate tokens in question (rough estimate)
                tokens = len(q["question"].split()) * 1.3
                cost = tokens * cfg["openai"][0]["pricing_per_1k_tokens"] / 1000
            
            # Find where the golden chunk appears
            found = False
            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}
                chunk_id = payload.get("chunk_id", hit.id)
                
                if chunk_id == q["gold_chunk_id"]:
                    found = True
                    metrics["overall"]["found_in_top_k"] += 1
                    metrics["by_strategy"][strategy]["found_in_top_k"] += 1
                    
                    # Update rank distribution
                    metrics["overall"]["rank_distribution"][rank] = metrics["overall"]["rank_distribution"].get(rank, 0) + 1
                    metrics["by_strategy"][strategy]["rank_distribution"][rank] = \
                        metrics["by_strategy"][strategy]["rank_distribution"].get(rank, 0) + 1
                    
                    # Update MRR
                    metrics["overall"]["mean_reciprocal_rank"] += 1.0 / rank
                    metrics["by_strategy"][strategy]["mean_reciprocal_rank"] += 1.0 / rank
                    
                    # Add the stored latency from embedding
                    chunk_latency = payload.get("latency", 0)
                    metrics["overall"]["total_latency_ms"] += chunk_latency
                    metrics["by_strategy"][strategy]["total_latency_ms"] += chunk_latency
                    
                    # Add the stored cost from embedding
                    chunk_cost = payload.get("cost", 0)
                    metrics["overall"]["total_cost"] += chunk_cost
                    metrics["by_strategy"][strategy]["total_cost"] += chunk_cost
                    
                    print(f"✓ Question: {q['question']}")
                    print(f"  Found golden chunk at rank {rank}")
                    print(f"  Chunk latency: {chunk_latency:.1f}ms")
                    print(f"  Chunk cost: ${chunk_cost:.4f}")
                    break
            
            if not found:
                print(f"✗ Question: {q['question']}")
                print(f"  Golden chunk not found in top {top_k} results")
    
    # Calculate final metrics
    total_questions = metrics["overall"]["total_questions"]
    if total_questions > 0:
        metrics["overall"]["recall_at_k"] = metrics["overall"]["found_in_top_k"] / total_questions
        metrics["overall"]["mean_reciprocal_rank"] /= total_questions
        metrics["overall"]["avg_latency_ms"] = metrics["overall"]["total_latency_ms"] / total_questions
        
        for strategy in metrics["by_strategy"]:
            strategy_total = metrics["by_strategy"][strategy]["total"]
            if strategy_total > 0:
                metrics["by_strategy"][strategy]["recall_at_k"] = \
                    metrics["by_strategy"][strategy]["found_in_top_k"] / strategy_total
                metrics["by_strategy"][strategy]["mean_reciprocal_rank"] /= strategy_total
                metrics["by_strategy"][strategy]["avg_latency_ms"] = \
                    metrics["by_strategy"][strategy]["total_latency_ms"] / strategy_total
    
    # Save metrics to logs directory
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get the strategy from the first question (all questions in a file use the same strategy)
    strategy = next(iter(metrics["by_strategy"].keys())) if metrics["by_strategy"] else cfg["strats"][0]
    model_name = model_name.replace("/", "_")  # Replace slashes with underscores for filename safety
    output_file = os.path.join(logs_dir, f"{strategy}_{model_name}_log.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n=== Retrieval Evaluation Results ===")
    print(f"Total questions evaluated: {metrics['overall']['total_questions']}")
    print(f"Recall@{top_k}: {metrics['overall']['recall_at_k']*100:.1f}%")
    print(f"Mean Reciprocal Rank: {metrics['overall']['mean_reciprocal_rank']:.3f}")
    print(f"Average embedding latency: {metrics['overall']['avg_latency_ms']:.1f}ms")
    print(f"Total embedding cost: ${metrics['overall']['total_cost']:.4f}")
    
    print("\nResults by Chunking Strategy:")
    for strategy, stats in metrics["by_strategy"].items():
        print(f"\n{strategy}:")
        print(f"  Total questions: {stats['total']}")
        print(f"  Recall@{top_k}: {stats['recall_at_k']*100:.1f}%")
        print(f"  Mean Reciprocal Rank: {stats['mean_reciprocal_rank']:.3f}")
        print(f"  Average embedding latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"  Total embedding cost: ${stats['total_cost']:.4f}")
    
    print(f"\nDetailed metrics saved to: {output_file}")

if __name__ == "__main__":
    evaluate_retrieval() 