
def embed(chunk, config):
        for model in config["embedding"]:
            print(model["provider"])
            print(chunk["chunk_id"])