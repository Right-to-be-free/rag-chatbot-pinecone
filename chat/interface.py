import requests
import streamlit as st

# Load secrets securely
hf_token = st.secrets["api"]["hf_token"]

# Use correct HF inference endpoint
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

# Authentication header
headers = {
    "Authorization": f"Bearer {hf_token}"
}

# Call Hugging Face Inference API
def generate_from_api(prompt: str, max_tokens=256):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # Display specific errors in the Streamlit UI
        if response.status_code != 200:
            st.error(f"❌ HF API Error {response.status_code}: {response.text}")
            raise Exception(f"HF API error {response.status_code}")

        json_response = response.json()

        # Handle unexpected formats
        if isinstance(json_response, list) and "generated_text" in json_response[0]:
            return json_response[0]["generated_text"]
        else:
            st.error("⚠️ Unexpected API response format.")
            return "[No answer generated]"

    except Exception as e:
        st.error(f"🚨 Exception: {str(e)}")
        return "[Error occurred while generating response]"

# LLMInterface wrapper class
class LLMInterface:
    def ask(self, question: str, context: str = "") -> str:
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:"""
        return generate_from_api(prompt)
