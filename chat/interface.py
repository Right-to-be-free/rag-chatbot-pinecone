import requests
import streamlit as st

# Load API key securely
together_api_key = st.secrets["api"]["together_api_key"]

# Together AI endpoint and model
API_URL = "https://api.together.xyz/v1/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Headers
headers = {
    "Authorization": f"Bearer {together_api_key}",
    "Content-Type": "application/json"
}

def generate_from_api(prompt: str, max_tokens=256):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stop": ["\n\n"]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            st.error(f"❌ Together API Error {response.status_code}: {response.text}")
            raise Exception(f"Together API error {response.status_code}")

        data = response.json()
        return data["choices"][0]["text"].strip()

    except Exception as e:
        st.error(f"🚨 Exception: {str(e)}")
        return "[Error occurred while generating response]"

# Wrapper class
class LLMInterface:
    def ask(self, question: str, context: str = "") -> str:
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:"""
        return generate_from_api(prompt)
