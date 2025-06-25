import os
import streamlit as st
import openai

# Load API key from Streamlit secrets or environment variable
openai.api_key = st.secrets["api"]["openai_api_key"]

MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4" if you have access

def generate_from_api(prompt: str):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return "⚠️ LLM response error. Please check OpenAI API key or model name."


class LLMInterface:
    def ask(self, question: str, context: str = ""):
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

        print("\n📝 Prompt to OpenAI Chat Model:\n", prompt)

        result = generate_from_api(prompt)
        print("\n✅ LLM Output:\n", result)
        return result
