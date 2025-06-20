import os
import json
import sys
import argparse
from datetime import datetime
import logging
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

from src.embedding import OpenAIEmbedder, HFEmbedder
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

def evaluate_retrieval(pairs: dict, user_id: str, provider: str, model: str, strategy: str):
    """Evaluate retrieval performance using files stored in Supabase."""
    try:
        if provider == "openai":
            embedder = OpenAIEmbedder(model, os.getenv("OPENAI_API_KEY"))
        elif provider in ("huggingface", "hf"):
            embedder = HFEmbedder(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Get retrieval parameters
        top_k = 5
        
        # Initialize metrics collection
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "config": {
                "embedding_provider": provider,
                "embedding_model": model,
                "chunking_strategy": strategy,
                "top_k": top_k
            },
            "stats": {
                "found_in_top_k": 0,
                "total_latency_ms": 0,
                "total_cost": 0,
                "rank_distribution": {},
                "recall_at_k": 0.0,
                "mean_reciprocal_rank": 0.0
            }
        }
        
        # Get all QA pair files for the user

            
        for pair in pairs:

            query_vector = embedder.embed(pair["question"])
            collection_name = f"autoembed_chunks_{user_id}"
            hits = search(query_vector, collection_name, top_k)
            
            # Log the top 5 chunks and their scores
            logger.info(f"Top {len(hits)} chunks for question: {pair['question']}")
            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}
                chunk_id = payload.get("chunk_id", hit.id)
                score = hit.score
                logger.info(f"  Rank {rank}: {chunk_id} (Score: {score:.4f})")
            
            # Calculate cost (if using OpenAI)
            cost = 0
            # Find where the golden chunk appears
            found = False
            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}
                chunk_id = payload.get("chunk_id", hit.id)
                
                if chunk_id == pair["gold_chunk_id"]:
                    found = True
                    metrics["stats"]["found_in_top_k"] += 1
                    metrics["stats"]["rank_distribution"][rank] = metrics["stats"]["rank_distribution"].get(rank, 0) + 1
                    
                    # Update MRR
                    metrics["stats"]["mean_reciprocal_rank"] += 1.0 / rank
                    
                    # Add the stored latency from embedding
                    chunk_latency = payload.get("latency", 0)
                    metrics["stats"]["total_latency_ms"] += chunk_latency
                    
                    # Add the stored cost from embedding
                    chunk_cost = payload.get("cost", 0)
                    metrics["stats"]["total_cost"] += chunk_cost
                    
                    logger.info(f"✓ Question: {pair['question']}")
                    logger.info(f"  Found golden chunk at rank {rank}")
                    logger.info(f"  Chunk latency: {chunk_latency:.1f}ms")
                    logger.info(f"  Chunk cost: ${chunk_cost:.4f}")
                    break
            
            if not found:
                logger.info(f"✗ Question: {pair['question']}")
                logger.info(f"  Golden chunk not found in top {top_k} results")
        
        # Calculate final metrics
        metrics["stats"]["recall_at_k"] = metrics["stats"]["found_in_top_k"] / 5
        metrics["stats"]["mean_reciprocal_rank"] /= 5
            
            # Save metrics to Supabase
            
            # Print summary
        logger.info("\n=== Retrieval Evaluation Results ===")
        logger.info(f"Recall@{top_k}: {metrics['stats']['recall_at_k']*100:.1f}%")
        logger.info(f"Mean Reciprocal Rank: {metrics['stats']['mean_reciprocal_rank']:.3f}")
        logger.info(f"Total embedding cost: ${metrics['stats']['total_cost']:.4f}")
            
        logger.info(f"\nChunking Strategy: {strategy}")
        logger.info(f"Embedding Provider: {provider}")
        logger.info(f"Embedding Model: {model}")
            
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