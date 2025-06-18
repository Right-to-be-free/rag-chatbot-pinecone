from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str):
        return self.model.encode(text).tolist()

    def embed_texts(self, texts: list):
        return self.model.encode(texts).tolist()