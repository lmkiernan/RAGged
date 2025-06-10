import os
import json
from src.embedding import embed
from src.config import load_config

chunks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chunks'))

chunk_files = {fn for fn in os.listdir(chunks_dir) if fn.endswith('.json')}

config = load_config("config/default.yaml")

for fn in chunk_files:
    with open(os.path.join(chunks_dir, fn), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        for chunk in chunks:
            embed(chunk, config)


