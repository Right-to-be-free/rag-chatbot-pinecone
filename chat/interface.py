import requests
import streamlit as st

# Secure Hugging Face token via Streamlit secrets
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

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError if not 200

        data = response.json()
        print("🔁 Raw API Response:", data)

        # Validate structure of the response
        if isinstance(data, list) and 'generated_text' in data[0]:
            return data[0]['generated_text']
        else:
            raise ValueError("Unexpected response format from Hugging Face API.")

    except Exception as e:
        print(f"❌ Error while calling LLM API: {e}")
        return "⚠️ LLM response error. Check your prompt, context, or API setup."


class LLMInterface:
    def ask(self, question: str, context: str = ""):
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:"""

        print("\n📝 Final Prompt Sent to LLM:\n", prompt)

        result = generate_from_api(prompt)

        print("\n✅ LLM Output:\n", result)
        return result
