import qdrant_client
from qdrant_client import QdrantClient
import uuid

client = QdrantClient(url="http://localhost:6333")


def uuid_from_string(s: str) -> str:
    # Use a namespace (here, DNS is common, but you can use your own)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

# Example payload
def upsert_vector(vector: list[float], payload: dict, id: str):
    client.upsert(
        collection_name="autoembed_chunks",
        points=[{
            "id": uuid_from_string(id),
            "vector": vector,
            "payload": payload
        }]
    )

def search(query_vector: list[float], top_k: int = 5, filter: dict = None):

    return client.search(
        collection_name="autoembed_chunks",
        query_vector=query_vector,
        limit=top_k,
        query_filter=filter
    )

