import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embedding_router import embed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_embeddings(chunk_list: list[dict], user_id: str):
    for chunk in chunk_list:
        logger.info(f"Embedding chunk: {chunk['chunk_id']}")
        embed(chunk, user_id)

