import os
import glob
import json
from src.chunk_router import chunk
from src.config import load_config

INGESTED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ingested'))

model_provider_map = { 
    "BAAI/bge-large-en": "huggingface"
}

config = load_config("config/default.yaml")

for filename in os.listdir(INGESTED_DIR):
    file_path = os.path.join(INGESTED_DIR, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    chunk(doc["text"], filename, config, model_provider_map)