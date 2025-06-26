import os
import streamlit as st
import requests

# ────────────────────────────────────────────────────────────────
# API Key Setup
# ────────────────────────────────────────────────────────────────
TOGETHER_API_KEY = (
    st.secrets.get("api", {}).get("together_api_key")
    or os.getenv("TOGETHER_API_KEY")
)

if not TOGETHER_API_KEY:
    st.error("❌ Together API key not found – add it to .streamlit/secrets.toml")
    st.stop()

# ────────────────────────────────────────────────────────────────
# Together API Configuration
# ────────────────────────────────────────────────────────────────
API_URL = "https://api.together.xyz/v1/chat/completions"  # ✅ Chat endpoint
MODEL_NAME = "togethercomputer/llama-2-7b-chat"            # ✅ Chat-supported model

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

# ────────────────────────────────────────────────────────────────
# Response Generator
# ────────────────────────────────────────────────────────────────
def generate_response(prompt: str, max_tokens: int = 256):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    # Optional debug logs
    print("🔁 Together API response:", response.status_code)
    print(response.text)

    if response.status_code != 200:
        raise Exception(f"Together API error: {response.status_code} – {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()

# ────────────────────────────────────────────────────────────────
# LLM Interface Wrapper
# ────────────────────────────────────────────────────────────────
class LLMInterface:
    def ask(self, question: str, context: str = "") -> str:
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
        return generate_response(prompt)
