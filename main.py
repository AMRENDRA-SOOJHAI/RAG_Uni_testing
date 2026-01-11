########################################################--------------------
import os
import numpy as np
# from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langfuse.openai import OpenAI

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# KNOWLEDGE BASE
# DOCUMENTS = [
#     "RAG stands for Retrieval Augmented Generation.",
#     "RAG uses embeddings to find relevant documents.",
#     "LLMs generate answers using provided context."
# ]

def load_documents(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

DOCUMENTS = load_documents("input.txt")

# EMBEDDING FUNCTION
def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in res.data])


# PRE-COMPUTE DOCUMENT EMBEDDINGS
DOC_EMBEDDINGS = embed(DOCUMENTS)


# RETRIEVER
def retrieve(query, k=2):
    query_emb = embed([query])
    scores = cosine_similarity(query_emb, DOC_EMBEDDINGS)[0]

    top_idx = scores.argsort()[-k:][::-1]
    return [DOCUMENTS[i] for i in top_idx]


# main RAG function
def rag(question):
    context = "\n".join(retrieve(question))

    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content
