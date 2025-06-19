import os
import json
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load configuration
try:
    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')) as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    raise

# Initialize Qdrant client
try:
    client = QdrantClient(
        url=config["qdrant"]["url"],
        api_key=config["qdrant"]["api_key"]
    )
except Exception as e:
    logger.error(f"Error initializing Qdrant client: {str(e)}")
    raise

def uuid_from_string(s: str) -> str:
    # Use a namespace (here, DNS is common, but you can use your own)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def create_collection(collection_name: str, vector_size: int = 1536):
    """Create a new collection for a user if it doesn't exist."""
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            # Delete existing collection to ensure fresh start
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Recreated collection: {collection_name}")
            
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {str(e)}")
        raise

def upsert_vector(vector, payload, id, collection_name: str):
    """Upsert a vector into the specified collection."""
    try:
        # Ensure collection exists (and is fresh)
        create_collection(collection_name)
        
        # Upsert the vector
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info(f"Successfully upserted vector {id} into collection {collection_name}")
        
    except Exception as e:
        logger.error(f"Error upserting vector {id} into collection {collection_name}: {str(e)}")
        raise

def search_vectors(query_vector, collection_name: str, limit: int = 5):
    """Search for similar vectors in the specified collection."""
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        logger.info(f"Successfully searched collection {collection_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error searching collection {collection_name}: {str(e)}")
        raise

def delete_collection(collection_name: str):
    """Delete a collection."""
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Successfully deleted collection {collection_name}")
        
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        raise

