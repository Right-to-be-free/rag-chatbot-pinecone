from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str):
        # Force CPU to avoid NotImplementedError from .to(device)
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str):
        return self.model.encode(text, device='cpu').tolist()

    def embed_texts(self, texts: list):
        return self.model.encode(texts, device='cpu').tolist()
