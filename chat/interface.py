import os
import streamlit as st
from together import Together

# Load API key from Streamlit secrets or environment variable
together_api_key = st.secrets["api"]["together_api_key"]
os.environ["TOGETHER_API_KEY"] = together_api_key  # Needed by Together()

# Initialize Together SDK client
client = Together()

MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # ✅ Use any Together-supported chat model

def generate_from_api(prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Together SDK Error:", e)
        return "⚠️ LLM response error. Please check Together API key or model name."


class LLMInterface:
    def ask(self, question: str, context: str = ""):
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

        print("\n📝 Prompt to Together Chat Model:\n", prompt)

        result = generate_from_api(prompt)
        print("\n✅ LLM Output:\n", result)
        return result
