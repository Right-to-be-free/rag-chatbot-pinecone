# llm_interface.py
import os
import streamlit as st
import openai

# ───────────────────────────────────────────────────────────────────────────────
#  API-key handling
# ───────────────────────────────────────────────────────────────────────────────
openai.api_key = (
    st.secrets.get("api", {}).get("openai_api_key")     # Streamlit Cloud
    or os.getenv("OPENAI_API_KEY")                      # local dev / CI
)

if not openai.api_key:
    st.error("❌ OpenAI key not found – add it to .streamlit/secrets.toml")
    st.stop()

MODEL_NAME   = "gpt-3.5-turbo"   # change to "gpt-4o-mini" etc. if you have access
MAX_TOKENS   = 1024
TEMPERATURE  = 0.7


# ───────────────────────────────────────────────────────────────────────────────
#  Low-level helper
# ───────────────────────────────────────────────────────────────────────────────
def _chat_complete(prompt: str) -> str:
    """
    Call the OpenAI Chat API (>=1.0 syntax) and return the assistant's reply.
    """
    try:
        resp = openai.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            timeout     = 30,          # avoid hanging forever
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        # Log traceback to Streamlit Cloud console
        import traceback, sys
        traceback.print_exc(file=sys.stderr)

        st.error(f"❌ OpenAI API Error: {e}")
        return "⚠️  LLM response error – check key, model name, or usage limits."


# ───────────────────────────────────────────────────────────────────────────────
#  Public interface
# ───────────────────────────────────────────────────────────────────────────────
class LLMInterface:
    """
    Thin wrapper that builds a nice prompt and calls _chat_complete().
    """

    def ask(self, question: str, context: str = "") -> str:
        prompt = (
            "You are a helpful assistant. Use the following context to answer "
            "the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        # Uncomment for debugging:
        # st.write("📝 Prompt preview:", prompt[:800])

        return _chat_complete(prompt)
