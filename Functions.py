from langchain_core.messages.chat import ChatMessage
import streamlit as st


def format_document_dynamically(docs):
    formatted_strings = []
    for i, doc in enumerate(docs):
        result = f"--- Document {i+1} ---\nContent: {doc.page_content}"
        metadata_str = "\nMetadata: " + " ,".join(
            [f"{key}:{value}" for key, value in doc.metadata.items()]
        )
        result += metadata_str

        formatted_strings.append(result)

    return "\n\n".join(formatted_strings)


def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))
