import requests
import streamlit as st

together_api_key = st.secrets["api"]["together_api_key"]  # ✅ Updated for Together
API_URL = "https://api.together.xyz/v1/completions"        # ✅ Together's endpoint

headers = {
    "Authorization": f"Bearer {together_api_key}",
    "Content-Type": "application/json"
}

MODEL_NAME = "togethercomputer/zephyr-7b-alpha"  # or any other Together model you prefer

def generate_from_api(prompt: str, max_tokens=256):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["\n\n"]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print("🔁 Together API raw response:", result)

        return result['choices'][0]['text'].strip()

    except Exception as e:
        print("❌ Together API Error:", e)
        return "⚠️ LLM response error. Please check Together API or model name."


class LLMInterface:
    def ask(self, question: str, context: str = ""):
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:"""

        print("\n📝 Prompt to Together API:\n", prompt)

        result = generate_from_api(prompt)
        print("\n✅ Together LLM Output:\n", result)
        return result
