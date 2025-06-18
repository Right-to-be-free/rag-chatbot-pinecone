import requests
import streamlit as st

hf_token = st.secrets["api"]["hf_token"]
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

headers = {
    "Authorization": f"Bearer {hf_token}"
}

def generate_from_api(prompt: str, max_tokens=256):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.json()[0]['generated_text']


class LLMInterface:
    def ask(self, question: str, context: str = ""):
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:"""
        return generate_from_api(prompt)
