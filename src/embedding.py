
def embed(chunk, config):
        for model in config["embedding"]:
            if model["provider"] == "openai":
                 embed_openai(chunk, model)

def embed_openai(chunk, config):
    print(chunk["chunk_id"])

def embed_huggingface(chunk, config):
    print(chunk["chunk_id"])