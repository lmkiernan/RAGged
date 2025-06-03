#Loads and validates the YAML configuration (e.g., which chunking strategy to use, token sizes, overlap). 
#Exposes those settings to the ingestion and chunking code.
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    