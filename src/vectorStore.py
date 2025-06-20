import logging
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load configuration



# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API")
)

def uuid_from_string(s: str) -> str:
    # Use a namespace (here, DNS is common, but you can use your own)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def ensure_collection_exists(collection_name: str, vector_size: int = 1536):
    """Ensure a collection exists, create it if it doesn't."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Successfully created collection {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
            
    except Exception as e:
        logger.error(f"Error ensuring collection exists {collection_name}: {str(e)}")
        raise

def upsert_vector(vector, payload, id, collection_name: str):
    """Upsert a vector into the specified collection."""
    try:
        # Ensure collection exists before upserting
        ensure_collection_exists(collection_name)
        
        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": uuid_from_string(id),
                "vector": vector,
                "payload": payload
            }]
        )
        logger.info(f"Successfully upserted vector {id} into collection {collection_name}")
        
    except Exception as e:
        logger.error(f"Error upserting vector {id} into collection {collection_name}: {str(e)}")
        raise

def search(query_vector, collection_name: str, limit: int = 5):
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

