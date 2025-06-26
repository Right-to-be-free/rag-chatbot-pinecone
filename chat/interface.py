import os
import streamlit as st
import requests

# Use Together API key
TOGETHER_API_KEY = (
    st.secrets.get("api", {}).get("together_api_key") or os.getenv("TOGETHER_API_KEY")
)

if not TOGETHER_API_KEY:
    st.error("❌ Together API key not found – add it to .streamlit/secrets.toml")
    st.stop()

API_URL = "https://api.together.xyz/inference/v1/completions"  # Example, change to your model's actual endpoint

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

def generate_response(prompt: str, max_tokens: int = 256):
    payload = {
        "model": "togethercomputer/llama-2-7b-chat",  # replace with your model
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Together API error: {response.status_code} – {response.text}")

    return response.json()["choices"][0]["text"]
