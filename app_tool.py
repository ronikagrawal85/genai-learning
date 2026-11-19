from agentic_chatbot_tool_backend import chatbot , get_all_threads
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import json
import streamlit as st
import uuid


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())



# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)



# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])



# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])



# Return only the text a person should see
def message_text(message):

    content = message.content

    # Plain string replies can be shown directly
    if isinstance(content, str):
        return content

    # Some models return a list of content blocks
    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") != "reasoning":
                parts.append(str(block.get("text") or ""))

        return "".join(parts)

    return ""



# Turn tool arguments into one readable line
def format_tool_args(args):

    if isinstance(args, str):
        return args

    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)

    return str(args or "")



# Draw one tool call in the chat
def render_tool_call(name, args, result=""):

    label = f"Calling tool: {name or 'tool'}"

    if args:
        label = f"{label}\n{args}"

    # A result means the tool has finished
    if result:
        st.success(label.replace("Calling tool:", "Used tool:", 1))

        with st.expander("Tool result"):
            st.text(result)
    else:
        st.info(label)



# Display the main application title
st.title("Agentic Chatbot with LangGraph")


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()



# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()




# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Match each tool result to the tool call that requested it
        tool_calls_by_id = {}


        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                temp_messages.append({
                    "role": "user",
                    "content": message_text(message)
                })

            # An assistant message may request one or more tools
            elif isinstance(message, AIMessage):

                for tool_call in message.tool_calls or []:
                    tool_entry = {
                        "role": "tool",
                        "name": tool_call.get("name", "tool"),
                        "args": format_tool_args(tool_call.get("args", "")),
                        "result": ""
                    }
                    temp_messages.append(tool_entry)

                    if tool_call.get("id"):
                        tool_calls_by_id[tool_call["id"]] = tool_entry

                # Show the written answer only when the model produced one
                answer = message_text(message).strip()

                if answer:
                    temp_messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            # Attach the tool output to the call that produced it
            elif isinstance(message, ToolMessage):
                tool_entry = tool_calls_by_id.get(message.tool_call_id)

                if tool_entry is None:
                    tool_entry = {
                        "role": "tool",
                        "name": message.name or "tool",
                        "args": "",
                        "result": message_text(message)
                    }
                    temp_messages.append(tool_entry)
                else:
                    tool_entry["result"] = message_text(message)

                    if message.name:
                        tool_entry["name"] = message.name


        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()



# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Tool calls use the assistant avatar, with their own status block
    if message["role"] == "tool":
        with st.chat_message("assistant"):
            render_tool_call(
                message.get("name", "tool"),
                message.get("args", ""),
                message.get("result", "")
            )
        continue

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])



# Create the chat input box
user_input = st.chat_input("Type here")


# Run this block after the user submits a message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    # Display the user's message in the chat interface
    })


    with st.chat_message("user"):
        st.text(user_input)


    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    # CONFIG = {
    #     "configurable": {
    #         "thread_id": st.session_state["thread_id"]
    #     }
    # }

    # to trace in langsmit metadata is used
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }



    # Tool calls seen during this turn, in the order they started
    tool_calls = {}
    tool_order = []

    # The final written answer, built from assistant text chunks
    ai_message = ""
    answer_box = None


    # Replace the contents of one tool slot as its call streams in
    def show_tool_call(tool_call):

        with tool_call["slot"].container():
            render_tool_call(
                tool_call["name"],
                tool_call["args"],
                tool_call["result"]
            )


    # Create the assistant chat-message container
    with st.chat_message("assistant"):

        # Stream messages from the LangGraph chatbot
        for message_chunk, metadata in chatbot.stream(
            {
                # Send the latest user message to the chatbot
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },

            # Use the current conversation thread
            config=CONFIG,

            # Stream individual message chunks
            stream_mode="messages"
        ):

            # Show the tool output as soon as the tool node returns it
            if isinstance(message_chunk, ToolMessage):
                matched_call = None

                for tool_index in tool_order:
                    tool_call = tool_calls[tool_index]

                    if (
                        message_chunk.tool_call_id
                        and tool_call.get("id") == message_chunk.tool_call_id
                    ):
                        matched_call = tool_call
                        break

                # Fall back to the latest unfinished call with the same name
                if matched_call is None:
                    for tool_index in reversed(tool_order):
                        tool_call = tool_calls[tool_index]

                        if tool_call["result"]:
                            continue

                        if (
                            not message_chunk.name
                            or tool_call["name"] == message_chunk.name
                        ):
                            matched_call = tool_call
                            break

                if matched_call is None:
                    matched_call = {
                        "name": message_chunk.name or "tool",
                        "args": "",
                        "id": message_chunk.tool_call_id,
                        "result": "",
                        "slot": st.empty()
                    }
                    tool_index = f"result-{len(tool_order)}"
                    tool_calls[tool_index] = matched_call
                    tool_order.append(tool_index)

                matched_call["result"] = message_text(message_chunk)

                if message_chunk.name:
                    matched_call["name"] = message_chunk.name

                show_tool_call(matched_call)
                continue

            # Ignore user messages echoed by the stream
            if not isinstance(message_chunk, AIMessage):
                continue

            # A tool request arrives in pieces: name, then argument text
            streamed_tool_calls = list(
                getattr(message_chunk, "tool_call_chunks", None) or []
            )

            # Some chunks already contain a finished tool call
            if not streamed_tool_calls:
                for tool_call in message_chunk.tool_calls or []:
                    streamed_tool_calls.append({
                        "index": tool_call.get("id") or len(tool_order),
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "args": format_tool_args(tool_call.get("args", ""))
                    })

            for tool_chunk in streamed_tool_calls:
                tool_index = tool_chunk.get("index")

                if tool_index is None:
                    tool_index = 0

                if tool_index not in tool_calls:
                    tool_calls[tool_index] = {
                        "name": "",
                        "args": "",
                        "id": None,
                        "result": "",
                        "slot": st.empty()
                    }
                    tool_order.append(tool_index)

                tool_call = tool_calls[tool_index]
                tool_name = tool_chunk.get("name") or ""

                # The name can arrive in one piece or a few pieces
                if tool_name:
                    if (
                        not tool_call["name"]
                        or tool_name.startswith(tool_call["name"])
                    ):
                        tool_call["name"] = tool_name
                    elif not tool_call["name"].endswith(tool_name):
                        tool_call["name"] += tool_name

                if tool_chunk.get("id"):
                    tool_call["id"] = tool_chunk["id"]

                if tool_chunk.get("args"):
                    tool_call["args"] += tool_chunk["args"]

                show_tool_call(tool_call)

            # Append only the visible answer, not the tool-call payload
            answer_piece = message_text(message_chunk)

            if answer_piece:
                ai_message += answer_piece

                if answer_box is None:
                    answer_box = st.empty()

                answer_box.text(ai_message)


    # Keep each tool call visible after the next Streamlit rerun
    for tool_index in tool_order:
        tool_call = tool_calls[tool_index]

        st.session_state["message_history"].append({
            "role": "tool",
            "name": tool_call["name"] or "tool",
            "args": tool_call["args"],
            "result": tool_call["result"]
        })

    # Save the complete assistant response in Streamlit session state
    if ai_message.strip():
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })