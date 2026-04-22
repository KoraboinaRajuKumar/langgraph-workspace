
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = ""

llm = init_chat_model("gpt-5.2")
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class greetState(BaseModel):
    messages: Annotated[list, add_messages]

# ---- GUARDRAIL FUNCTION ----
def guardrail_check(message: str):
    blocked_words = ["hack", "attack", "illegal"]
    
    for word in blocked_words:
        if word in message.lower():
            return False
    return True

# ---- NODE ----
def chatbotnode(state: greetState):
    user_msg = state.messages[-1].content

    # 🔥 GUARDRAIL CHECK
    if not guardrail_check(user_msg):
        return {"messages": [AIMessage(content="❌ This request is not allowed bz of Guardrails!")]}

    response = llm.invoke(state.messages)
    return {"messages": [AIMessage(content=response.content)]}

# ---- GRAPH ----
graph = StateGraph(greetState)

graph.add_node("chatBot", chatbotnode)
graph.add_edge(START, "chatBot")
graph.add_edge("chatBot", END)

memory_saver = InMemorySaver()
final_graph = graph.compile(checkpointer=memory_saver)

config = {"configurable": {"thread_id": "1234"}}

# STEP 1
res=final_graph.invoke({
    "messages": [HumanMessage(content="how to hack a system")]
}, config=config)



print(res["messages"][-1].content)
❌ This request is not allowed bz of Guardrails!