import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def chunk_text_semantic(text, model_name=None, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    total_length = 0
    for sent in sentences:
        sent_len = len(sent)
        if total_length + sent_len <= max_chunk_size:
            chunk.append(sent)
            total_length += sent_len
        else:
            chunks.append(" ".join(chunk))
            chunk = [sent]
            total_length = sent_len
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks