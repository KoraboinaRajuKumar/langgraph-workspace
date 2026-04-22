# ============================================================
# FULL RAG CODE + ONLY ADDED:
# 1. MMR Retrieval
# 2. Reranking
# ============================================================

# pip install langgraph langchain langchain-openai
# pip install langchain-community pypdf python-dotenv sentence-transformers

# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import List, Any

from langgraph.graph import StateGraph, START, END

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

from sentence_transformers import CrossEncoder

# ============================================================
# LLM
# ============================================================

llm = init_chat_model("gpt-5.2")

# ============================================================
# LOAD PDF
# ============================================================

loader = PyPDFLoader("sample.pdf")
docs = loader.load()

# ============================================================
# CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(docs)

# ============================================================
# EMBEDDINGS + VECTOR DB
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=embeddings
)

# ============================================================
# RERANK MODEL
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ============================================================
# STATE
# ============================================================

class RagState(BaseModel):
    question: str
    documents: List[Any] = Field(default_factory=list)
    context: str = ""
    answer: str = ""

# ============================================================
# NODE 1 - RETRIEVE (MMR)
# ============================================================

def retrieve_node(state: RagState) -> RagState:

    docs = vector_store.max_marginal_relevance_search(
        query=state.question,
        k=8,
        fetch_k=15
    )

    state.documents = docs
    return state

# ============================================================
# NODE 2 - RERANK
# ============================================================

def rerank_node(state: RagState) -> RagState:

    pairs = []

    for doc in state.documents:
        pairs.append(
            (state.question, doc.page_content)
        )

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(scores, state.documents),
        key=lambda x: x[0],
        reverse=True
    )

    state.documents = [
        doc for score, doc in ranked_docs[:3]
    ]

    return state

# ============================================================
# NODE 3 - CONTEXT
# ============================================================

def context_node(state: RagState) -> RagState:

    context = ""

    for i, doc in enumerate(state.documents, start=1):

        file_name = doc.metadata.get("source", "Unknown.pdf")
        page_no = doc.metadata.get("page", 0) + 1

        context += f"""
Source {i}
File : {file_name}
Page : {page_no}

{doc.page_content}

"""

    state.context = context
    return state

# ============================================================
# NODE 4 - GENERATE
# ============================================================

def generate_node(state: RagState) -> RagState:

    prompt = f"""
Answer only from provided context.

Rules:
1. Be concise
2. Include citations
3. If answer not found say I don't know

Context:
{state.context}

Question:
{state.question}
"""

    res = llm.invoke(prompt)

    state.answer = res.content
    return state

# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(RagState)

graph.add_node("retrieve_node", retrieve_node)
graph.add_node("rerank_node", rerank_node)
graph.add_node("context_node", context_node)
graph.add_node("generate_node", generate_node)

graph.add_edge(START, "retrieve_node")
graph.add_edge("retrieve_node", "rerank_node")
graph.add_edge("rerank_node", "context_node")
graph.add_edge("context_node", "generate_node")
graph.add_edge("generate_node", END)

app = graph.compile()

# ============================================================
# RUN
# ============================================================

response = app.invoke({
    "question": "What baselines were used to compare Meta reward models?"
})

# ============================================================
# OUTPUT
# ============================================================

print("=" * 70)
print("QUESTION :")
print(response["question"])
print("=" * 70)

print("ANSWER :")
print(response["answer"])
print("=" * 70)

print("TOP SOURCES :")

for i, doc in enumerate(response["documents"], start=1):

    file_name = doc.metadata.get("source", "Unknown.pdf")
    page_no = doc.metadata.get("page", 0) + 1

    print(f"{i}. {file_name} | Page {page_no}")

print("=" * 70)