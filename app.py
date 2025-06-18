import streamlit as st
from document_manager import DocumentManager
from chat.interface import LLMInterface

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.sidebar.title("⚙️ Configuration")
db_type = "pinecone"
model_name = st.sidebar.selectbox("Select Embedding Model", [
    "all-MiniLM-L6-v2", "all-mpnet-base-v2", "distilbert-base-nli-stsb-mean-tokens"
])

llm = LLMInterface()
doc_manager = DocumentManager(db_type=db_type, model_name=model_name)

st.title("🧠 RAG-powered QA Chatbot")
user_query = st.text_input("Ask a question about your documents:")
submit_button = st.button("Submit")

if submit_button and user_query:
    with st.spinner("Retrieving relevant context..."):
        relevant_docs = doc_manager.query(user_query)
        if not relevant_docs:
            st.warning("No relevant documents found.")
        else:
            context = "\n\n".join([
                f"[{i+1}] {doc.get('metadata', {}).get('chunk_text', '')}"
                for i, doc in enumerate(relevant_docs)
            ])

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

            with st.spinner("Generating response..."):
                answer = llm.ask(user_query, context=context)
                st.success("💬 Answer:")
                st.write(answer)