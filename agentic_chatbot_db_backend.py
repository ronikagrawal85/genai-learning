from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # take user query from state
    messages = state["messages"]
    # send to llm
    response = llm.invoke(messages)
    # response store state
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)
