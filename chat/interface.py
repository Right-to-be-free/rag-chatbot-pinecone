import os
import streamlit as st
import openai

# 1. Load key
openai.api_key = st.secrets["api"]["openai_api_key"]

MODEL_NAME = "gpt-3.5-turbo"

def generate_from_api(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(        # ← new path
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"❌ OpenAI API Error: {e}")
        return "⚠️ LLM response error – check key, model name, or usage limits."

class LLMInterface:
    def ask(self, question: str, context: str = "") -> str:
        prompt = (
            "You are a helpful assistant. Use the following context to answer "
            "the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return generate_from_api(prompt)
