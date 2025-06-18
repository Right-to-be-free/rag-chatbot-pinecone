import requests
import streamlit as st

hf_token = st.secrets["api"]["hf_token"]
headers = {"Authorization": f"Bearer {hf_token}"}
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

class LLMInterface:
    def ask(self, query: str, context: str = "", max_tokens=256):
        prompt = f"Answer this question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "return_full_text": False}}
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")
        return response.json()[0]["generated_text"]