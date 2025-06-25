from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name: str):
        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Loading embedding model: {model_name} on device: {device}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str):
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]):
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
