import os, traceback, sys
import streamlit as st
import openai

# 1️⃣  Grab key from secrets *or* env var
openai.api_key = st.secrets.get("api", {}).get("openai_api_key") \
                 or os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("❌ OpenAI key not found – set it in Settings → Secrets")
    st.stop()

MODEL_NAME = "gpt-3.5-turbo"          # or "gpt-4o-mini" … etc.
MAX_TOKENS = 1024                     # keep eye on context length

def generate_from_api(prompt: str) -> str:
    try:
        # v1-style call; change to ChatCompletion.create if you’re on 0.28
        resp = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        # Surface the real traceback in Cloud logs *and* in the UI
        traceback.print_exc(file=sys.stderr)
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

        st.write("📝 **Prompt sent to OpenAI:**", prompt[:4000])  # optional debug
        return generate_from_api(prompt)
