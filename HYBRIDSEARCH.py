# app.py
# Single-file Production Ready RAG Demo
# Stack:
# LangChain + FAISS + OpenAI + MMR + Hybrid Search + ReRanking

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# =========================
# CONFIG
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PDF_PATH = "company_policy.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5


# =========================
# LOAD DOCUMENTS
# =========================

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


# =========================
# SPLIT DOCUMENTS
# =========================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    return chunks


# =========================
# CREATE VECTOR STORE
# =========================

def create_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db


# =========================
# KEYWORD SEARCH (BM25)
# =========================

def keyword_search(query, docs, k=TOP_K):
    corpus = [doc.page_content for doc in docs]

    tokenized = [text.split() for text in corpus]

    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(query.split())

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:k]]


# =========================
# SEMANTIC SEARCH + MMR
# =========================

def semantic_search(query, db):
    results = db.max_marginal_relevance_search(
        query,
        k=TOP_K,
        fetch_k=10
    )
    return results


# =========================
# HYBRID SEARCH
# =========================

def hybrid_search(query, db, docs):
    keyword_docs = keyword_search(query, docs)
    semantic_docs = semantic_search(query, db)

    merged = keyword_docs + semantic_docs

    final_docs = []
    seen = set()

    for doc in merged:
        text = doc.page_content

        if text not in seen:
            final_docs.append(doc)
            seen.add(text)

    return final_docs[:8]


# =========================
# RERANKING
# =========================

def rerank_documents(query, docs):
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = []

    for doc in docs:
        pairs.append([query, doc.page_content])

    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:4]]


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(query, docs):
    llm = ChatOpenAI(model="gpt-4o-mini")

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an enterprise AI assistant.

Answer only from the provided context.
If answer not found, say:
'I could not find this in the documents.'

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content


# =========================
# MAIN FLOW
# =========================

def run_rag():
    print("Loading PDF...")

    docs = load_documents(PDF_PATH)

    print("Splitting chunks...")

    chunks = split_documents(docs)

    print("Creating FAISS Index...")

    db = create_faiss_index(chunks)

    print("System Ready!")

    while True:
        query = input("\nAsk Question (or exit): ")

        if query.lower() == "exit":
            print("Goodbye")
            break

        retrieved_docs = hybrid_search(
            query,
            db,
            chunks
        )

        final_docs = rerank_documents(
            query,
            retrieved_docs
        )

        answer = generate_answer(
            query,
            final_docs
        )

        print("\nAnswer:")
        print(answer)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_rag()