# app.py

import os
import uuid
import streamlit as st

from typing import Annotated
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Memory Guardrail Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------------------------------
# CUSTOM UI
# ----------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.chat-box {
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.user-box {
    background-color: #1f77b4;
    color: white;
}
.bot-box {
    background-color: #262730;
    color: white;
}
.title {
    text-align:center;
    font-size:34px;
    font-weight:bold;
    color:#00d4ff;
}
.sub {
    text-align:center;
    color:gray;
    margin-bottom:25px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# API KEY
# ----------------------------------------------------
os.environ["OPENAI_API_KEY"] = ""

# ----------------------------------------------------
# LLM
# ----------------------------------------------------
llm = init_chat_model("gpt-5.2")

# ----------------------------------------------------
# STATE
# ----------------------------------------------------
class ChatState(BaseModel):
    messages: Annotated[list, add_messages]

# ----------------------------------------------------
# GUARDRAILS
# ----------------------------------------------------
def guardrail_check(message):
    blocked_words = [
        "hack",
        "attack",
        "illegal",
        "crack",
        "fraud"
    ]

    for word in blocked_words:
        if word in message.lower():
            return False
    return True

# ----------------------------------------------------
# CHATBOT NODE
# ----------------------------------------------------
def chatbot_node(state: ChatState):

    user_msg = state.messages[-1].content

    # Guardrail Check
    if not guardrail_check(user_msg):
        return {
            "messages": [
                AIMessage(
                    content="🚫 Sorry, this request is restricted by Guardrails."
                )
            ]
        }

    response = llm.invoke(state.messages)

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }

# ----------------------------------------------------
# GRAPH
# ----------------------------------------------------
graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

memory = InMemorySaver()

app_graph = graph.compile(checkpointer=memory)

# ----------------------------------------------------
# SESSION
# ----------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown(
    '<div class="title">🤖 Memory Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub">🧠 Powered by LangGraph | 🔐 Guardrails Enabled</div>',
    unsafe_allow_html=True
)

# ----------------------------------------------------
# INPUT
# ----------------------------------------------------
user_input = st.chat_input("💬 Ask me anything...")

if user_input:

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    result = app_graph.invoke(
        {"messages": st.session_state.messages},
        config=config
    )

    ai_reply = result["messages"][-1]

    st.session_state.messages.append(ai_reply)

# ----------------------------------------------------
# DISPLAY CHAT
# ----------------------------------------------------
for msg in st.session_state.messages:

    if isinstance(msg, HumanMessage):
        st.markdown(
            f'<div class="chat-box user-box">🧑 You: {msg.content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-box bot-box">🤖 Bot: {msg.content}</div>',
            unsafe_allow_html=True
        )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.caption("✅ Memory Enabled   |   🔒 Guardrails Active   |   ⚡ Streamlit UI")