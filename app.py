import streamlit as st
from document_manager import DocumentManager
from chat.interface import LLMInterface
import openai
openai.api_key = st.secrets["api"]["openai_api_key"]


st.write(response['choices'][0]['message']['content'])

# Page configuration
st.set_page_config(page_title="RAG Chatbot - Rishi's Assistant", layout="wide")

# Sidebar config
st.sidebar.title("⚙️ Configuration")
db_type = "pinecone"
model_name = st.sidebar.selectbox("Select Embedding Model", [
    "all-MiniLM-L6-v2", 
    "all-mpnet-base-v2", 
    "distilbert-base-nli-stsb-mean-tokens"
])

# Instantiate chatbot and doc manager
llm = LLMInterface()
doc_manager = DocumentManager(db_type=db_type, model_name=model_name)

# Title and intro
st.title("🧠 RAG-powered QA Chatbot")

st.markdown("""
### 👋 Welcome to **Rishi's GenAI Assistant**
I'm your smart assistant trained on:
- 🍛 **6,000+ Indian food recipes** — from snacks to desserts
- 🤖 **Generative AI** — transformers, LLMs, and applications
- 📡 **Apache Kafka** — streaming, architecture, real-world use cases

Ask me anything from *"How to make Hyderabadi Biryani?"* to *"Explain Kafka partitions"*.

---
""")

# Optional expandable info section
with st.expander("📁 About My Knowledge Base"):
    st.markdown("""
    - **Indian Recipes Dataset**: Regional, modern, and traditional recipes with ingredients and instructions.
    - **Generative AI Docs**: Collected insights on transformer models, embeddings, RAG, and LLM pipelines.
    - **Kafka Materials**: Notes on brokers, partitions, producers/consumers, and real-time data flows.
    """)

# User interaction
user_query = st.text_input("🔎 Ask a question about your documents:")
submit_button = st.button("Submit")

# Process input
if submit_button and user_query:
    with st.spinner("🔍 Retrieving relevant context..."):
        relevant_docs = doc_manager.query(user_query)

        if not relevant_docs:
            st.warning("🚫 No relevant documents found.")
        else:
            context = "\n\n".join([
                f"[{i+1}] {doc.get('metadata', {}).get('chunk_text', '')}"
                for i, doc in enumerate(relevant_docs)
            ])

            with st.expander("📄 Retrieved Context"):
                st.write(context)

            with st.spinner("✍️ Generating response..."):
                answer = llm.ask(user_query, context=context)
                st.success("💬 Answer:")
                st.write(answer)

# Footer
st.markdown("---")
st.markdown("✅ Powered by Pinecone + Streamlit | Built by **Rishi** 🚀")
