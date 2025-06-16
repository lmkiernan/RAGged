import os
import json
import sys
import argparse
from datetime import datetime
import logging
from supabase_client import SupabaseClient
from typing import List, Dict, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

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
            from .chunking.fixed_token import fixed_token_chunk
            return fixed_token_chunk(text, "temp", config, {}, "temp", model_name, provider)
        elif strategy == "sliding_window":
            from .chunking.sliding_window import sliding_window_chunk
            return sliding_window_chunk(text, "temp", config, {}, "temp", model_name, provider)
        elif strategy == "sentence_aware":
            from .chunking.sentence_aware import sentence_aware_chunk
            return sentence_aware_chunk(text, "temp", config, {}, "temp", model_name, provider)
        else:
            raise ValueError(f"Invalid chunking strategy: {strategy}")
    except ImportError as e:
        logger.error(f"Failed to import chunking module: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during chunking: {str(e)}")
        raise

def evaluate_retrieval(user_id: str):
    """Evaluate retrieval performance using files stored in Supabase."""
    try:
        # Initialize Supabase client
        supabase = SupabaseClient()
        
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
            "user_id": user_id,
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
        
        # Get all QA pair files for the user
        qa_files = supabase.list_files(user_id, prefix="qa_pairs/")
        if not qa_files:
            logger.warning(f"No QA pairs found for user {user_id}")
            return
            
        # Process each QA pair file
        for file_info in qa_files:
            file_path = file_info['name']
            if not file_path.endswith('_qa.json'):
                continue
                
            doc_id = os.path.basename(file_path).replace('_qa.json', '')
            logger.info(f"\nEvaluating questions for: {doc_id}")
            
            # Download QA pairs
            qa_data = supabase.download_file(file_path, user_id)
            if not qa_data:
                logger.warning(f"Failed to download QA pairs: {file_path}")
                continue
                
            questions = json.loads(qa_data.decode('utf-8'))
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
                        
                        logger.info(f"✓ Question: {q['question']}")
                        logger.info(f"  Found golden chunk at rank {rank}")
                        logger.info(f"  Chunk latency: {chunk_latency:.1f}ms")
                        logger.info(f"  Chunk cost: ${chunk_cost:.4f}")
                        break
                
                if not found:
                    logger.info(f"✗ Question: {q['question']}")
                    logger.info(f"  Golden chunk not found in top {top_k} results")
        
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
        
        # Save metrics to Supabase
        strategy = next(iter(metrics["by_strategy"].keys())) if metrics["by_strategy"] else cfg["strats"][0]
        model_name = model_name.replace("/", "_")  # Replace slashes with underscores for filename safety
        log_filename = f"{strategy}_{model_name}_log.json"
        storage_path = f"logs/{user_id}/{log_filename}"
        
        # Upload to Supabase
        result = supabase.supabase.storage.from_('documents').upload(
            storage_path,
            json.dumps(metrics, indent=2).encode('utf-8'),
            {'content-type': 'application/json'}
        )
        
        logger.info(f"Saved metrics to Supabase: {storage_path}")
        
        # Print summary
        logger.info("\n=== Retrieval Evaluation Results ===")
        logger.info(f"Total questions evaluated: {metrics['overall']['total_questions']}")
        logger.info(f"Recall@{top_k}: {metrics['overall']['recall_at_k']*100:.1f}%")
        logger.info(f"Mean Reciprocal Rank: {metrics['overall']['mean_reciprocal_rank']:.3f}")
        logger.info(f"Average embedding latency: {metrics['overall']['avg_latency_ms']:.1f}ms")
        logger.info(f"Total embedding cost: ${metrics['overall']['total_cost']:.4f}")
        
        logger.info("\nResults by Chunking Strategy:")
        for strategy, stats in metrics["by_strategy"].items():
            logger.info(f"\n{strategy}:")
            logger.info(f"  Total questions: {stats['total']}")
            logger.info(f"  Recall@{top_k}: {stats['recall_at_k']*100:.1f}%")
            logger.info(f"  Mean Reciprocal Rank: {stats['mean_reciprocal_rank']:.3f}")
            logger.info(f"  Average embedding latency: {stats['avg_latency_ms']:.1f}ms")
            logger.info(f"  Total embedding cost: ${stats['total_cost']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in evaluate_retrieval: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance')
    parser.add_argument('--user-id', required=True, help='User ID for storage')
    args = parser.parse_args()
    
    try:
        evaluate_retrieval(args.user_id)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 