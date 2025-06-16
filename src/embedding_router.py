import json
import os
import time
import logging
from src.Embedding import OpenAIEmbedder, HFEmbedder
from src.vectorStore import upsert_vector

# Configure logging
logging.basicConfig(level=logging.INFO)
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

def embed(chunk, config):
    """Generate embeddings for a chunk using configured models."""
    try:
        for model in config["embedding"]:
            if model["provider"] == "openai":
                embed_openai(chunk, config)
            elif model["provider"] == "huggingface":
                embed_huggingface(chunk, config)
    except Exception as e:
        logger.error(f"Error in embed function: {str(e)}")
        raise

def embed_openai(chunk, config):
    """Generate embeddings using OpenAI models."""
    try:
        for model in config["openai"]:
            embedder = get_embedder("openai", model["model"])
            t0 = time.time()
            vector = embedder.embed(chunk["text"])
            t1 = time.time()
            latency = (t1 - t0) * 1000
            
            # Calculate token count (rough estimate)
            token_count = len(chunk["text"].split()) * 1.3
            
            payload = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "strategy": chunk["strategy"],
                "user_id": chunk["user_id"],
                "token_count": token_count,
                "latency": latency,
                "cost": token_count * model["pricing_per_1k_tokens"] / 1000
            }
            
            # Add user_id to Qdrant collection name
            collection_name = f"autoembed_chunks_{chunk['user_id']}"
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
