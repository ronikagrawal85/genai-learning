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

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages'
            )
        )
        st.session_state['messages_history'].append({'role': 'assistant', 'content': ai_message})