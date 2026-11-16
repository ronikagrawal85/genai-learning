from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st


thread_id = "1"
config = {'configurable': {'thread_id': thread_id}}

st.title("GenAI Chatbot")

if "messages_history" not in st.session_state:
    st.session_state['messages_history'] = []

for message in st.session_state['messages_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Enter your message")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state['messages_history'].append({'role': 'user', 'content': user_input})

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
    with st.chat_message("assistant"):
        st.write(response['messages'][-1].content)
    st.session_state['messages_history'].append({'role': 'assistant', 'content': response['messages'][-1].content})
