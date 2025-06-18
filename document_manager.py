import os, time, json, hashlib, re
import numpy as np
from embedding_model import EmbeddingModel
from vector_db import PineconeVectorDB
from file_utils import load_file
from chunker import chunk_text_semantic

class DocumentManager:
    def __init__(self, db_type: str, model_name: str):
        self.embedding_model = EmbeddingModel(model_name)
        embed_dim = self.embedding_model.dim
        index_name = re.sub(r'[^a-z0-9\-]', '-', model_name.lower()) + f"-{embed_dim}"
        self.vector_db = PineconeVectorDB(index_name=index_name, dimension=embed_dim)
        self.meta_file = f"{index_name}_meta.json"
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                data = json.load(f)
                self.path_to_hash = data.get("path_to_hash", {})
                self.hash_to_id = data.get("hash_to_id", {})
        else:
            self.path_to_hash, self.hash_to_id = {}, {}

    def _save_metadata(self):
        with open(self.meta_file, "w") as f:
            json.dump({
                "path_to_hash": self.path_to_hash,
                "hash_to_id": self.hash_to_id
            }, f, indent=2)

    def ingest_file(self, file_path: str):
        file_path = os.path.abspath(file_path)
        content = load_file(file_path)
        if not content.strip(): return

        file_hash = hashlib.md5(content.encode()).hexdigest()
        if self.path_to_hash.get(file_path) == file_hash:
            return

        chunks = chunk_text_semantic(content, model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = self.embedding_model.embed_texts(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{file_path}_chunk{i}"
            metadata = {
                "file": file_path,
                "chunk_index": i,
                "chunk_text": chunk[:500],
                "hash": file_hash
            }
            self.vector_db.add_document(chunk_id, vec, metadata)
        self.path_to_hash[file_path] = file_hash
        self.hash_to_id[file_hash] = file_path
        self._save_metadata()

    def delete_document(self, file_path: str):
        self.vector_db.delete_document(file_path)
        self.path_to_hash.pop(file_path, None)
        self.hash_to_id.pop(file_path, None)
        self._save_metadata()

    def query(self, query_text: str, top_k: int = 5):
        embedding = self.embedding_model.embed_text(query_text)
        return self.vector_db.query(embedding, top_k=top_k)