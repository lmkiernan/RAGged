from abc import ABC, abstractmethod

token_cache = {}

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str, openai_key: str):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI is not installed. Please install it with `pip3 install openai`.")

        openai.api_key = openai_key
        self.client = openai
        self.model_name = model_name
    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        vector = response.data[0].embedding
        return vector

class HFEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch: pip install transformers torch")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, text: str) -> list[float]:
        from transformers import AutoTokenizer
        import torch
        cache_key = (self.model.config._name_or_path, text)
        if cache_key in token_cache:
            return token_cache[cache_key]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        token_cache[cache_key] = embedding
        return embedding
