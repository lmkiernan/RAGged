import logging
from src.embedding_router import embed
from config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_embeddings(chunk_list: list[dict]):
    for chunk in chunk_list:
        logger.info(f"Embedding chunk: {chunk['chunk_id']}")
        #embed(chunk)

