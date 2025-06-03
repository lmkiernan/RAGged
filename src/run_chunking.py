import os
import glob
import json
from src.chunk_router import chunk
from src.config import load_config

INGESTED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ingested'))

model_provider_map = { 
    "BAAI/bge-large-en": "huggingface"
}

os.makedirs("chunks", exist_ok=True)



config = load_config("config/default.yaml")

for filename in os.listdir(INGESTED_DIR):
    file_path = os.path.join(INGESTED_DIR, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    for strat in config["strats"]:
        chunks = chunk(doc["text"], filename, config, model_provider_map, strat)
        name_clean = os.path.splitext(filename)[0] 
        out_path = os.path.join("chunks", f"{name_clean}_{strat}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4)
