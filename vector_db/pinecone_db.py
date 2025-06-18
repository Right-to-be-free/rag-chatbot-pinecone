import streamlit as st
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorDB:
    def __init__(self, index_name: str, dimension: int):
        # ✅ Fetch credentials from Streamlit secrets
        api_key = st.secrets["api"]["pinecone_key"]
        env = st.secrets["api"]["pinecone_env"]

        if not api_key or not env:
            raise RuntimeError("PINECONE_API_KEY and PINECONE_ENV must be set in st.secrets.")

        self.dimension = dimension
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        # ✅ Check and create index if not exists
        index_names = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in index_names:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=env)
            )

        self.index = self.pc.Index(self.index_name)

    def add_document(self, doc_id, embedding, metadata=None):
        self.index.upsert(vectors=[(str(doc_id), embedding, metadata or {})])

    def delete_document(self, doc_id):
        self.index.delete(ids=[str(doc_id)])

    def query(self, embedding, top_k=5):
        result = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return result.get("matches", [])
