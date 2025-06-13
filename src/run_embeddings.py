import os
import json
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from embedding_router import embed
from config import load_config

chunks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chunks'))

chunk_files = {fn for fn in os.listdir(chunks_dir) if fn.endswith('.json')}

config = load_config("config/default.yaml")

for fn in chunk_files:
    with open(os.path.join(chunks_dir, fn), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        for chunk in chunks:
            embed(chunk, config)


