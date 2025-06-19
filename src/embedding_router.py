import os
import sys
import json
import argparse
from typing import List, Dict, Any
import logging
from src.supabase_client import SupabaseClient
import traceback
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Embedding import OpenAIEmbedder, HFEmbedder
from src.vectorStore import upsert_vector
from src.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embedding.log')
    ]
)
logger = logging.getLogger(__name__)

# Load API keys
try:
    with open(os.path.join(os.path.dirname(__file__), '..', 'APIKeys.json')) as f:
        api_keys = json.load(f)
    openai_key = api_keys.get('openai')
    huggingface_key = api_keys.get('huggingface')
except Exception as e:
    logger.error(f"Error loading API keys: {str(e)}")
    raise

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

def embed(chunk, user_id):
    """Generate embeddings for a chunk using configured models."""
    try:
        if chunk["provider"] == "openai":
                embed_openai(chunk, user_id)
        elif chunk["provider"] == "huggingface":
                embed_huggingface(chunk, user_id)
    except Exception as e:
        logger.error(f"Error in embed function: {str(e)}")
        raise

def embed_openai(chunk, user_id):
    """Generate embeddings using OpenAI models."""
    try:
        
        embedder = get_embedder("openai", chunk["model"])
        t0 = time.time()
        vector = embedder.embed(chunk["text"])
        t1 = time.time()
        latency = (t1 - t0) * 1000
            
        # Calculate token count (rough estimate)
        token_count = chunk["token_count"]

        config = load_config("config/default.yaml")
        price == 0
        for model in config["openai"]:
            if model["model"] == chunk["model"]:
                price = model["pricing_per_1k_tokens"]
                break
            
        payload = {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "strategy": chunk["strategy"],
            "token_count": token_count,
            "latency": latency,
            "cost": token_count * price / 1000
        }
            
        # Add user_id to Qdrant collection name
        collection_name = f"autoembed_chunks_{user_id}"
        upsert_vector(vector, payload, chunk["chunk_id"], collection_name)
        logger.info(f"Successfully embedded chunk {chunk['chunk_id']} using OpenAI {model['model']}")
            
    except Exception as e:
        logger.error(f"Error in embed_openai: {str(e)}")
        raise

def embed_huggingface(chunk, config):
    """Generate embeddings using HuggingFace models."""
    try:
        for model in config["huggingface"]:
            embedder = get_embedder("huggingface", model["model"])
            t0 = time.time()
            vector = embedder.embed(chunk["text"])
            t1 = time.time()
            latency = (t1 - t0) * 1000
            
            payload = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "strategy": chunk["strategy"],
                "user_id": chunk["user_id"],
                "latency": latency,
                "cost": 0  # HuggingFace models are free
            }
            
            # Add user_id to Qdrant collection name
            collection_name = f"autoembed_chunks_{chunk['user_id']}"
            upsert_vector(vector, payload, chunk["chunk_id"], collection_name)
            logger.info(f"Successfully embedded chunk {chunk['chunk_id']} using HuggingFace {model['model']}")
            
    except Exception as e:
        logger.error(f"Error in embed_huggingface: {str(e)}")
        raise

def get_embedder(provider: str, model_name: str, **kwargs):
    """Get an embedder instance for the specified provider and model."""
    try:
        provider = provider.lower()
        if provider == "openai":
            if not openai_key:
                raise ValueError("OpenAI API key not found")
            return OpenAIEmbedder(model_name, openai_key)
        elif provider in ("huggingface", "hf"):
            return HFEmbedder(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'huggingface'.")
    except Exception as e:
        logger.error(f"Error getting embedder: {str(e)}")
        raise
