# Import required packages
import streamlit as st
from streamlit_chat import message
import os
from backend.core import run_llm

# Page configuration
st.set_page_config(
    page_title="Optuna RAG Assistant",
    page_icon="🎯",
    layout="wide",
)

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/optuna-logo.png", width=100)
with col2:
    st.header("Optuna RAG Assistant")
    st.markdown("*Your AI-powered guide to Optuna optimization framework*")

# Create tabs for chat and documentation
tab1, tab2 = st.tabs(["Chat", "Reference Documents"])

with tab1:
    # Initialize session state
    if "chat_prompt_history" not in st.session_state:
        st.session_state["chat_prompt_history"] = []
    if "chat_answer_history" not in st.session_state:
        st.session_state["chat_answer_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "current_sources" not in st.session_state:
        st.session_state["current_sources"] = []

    # Chat interface
    st.markdown("### Ask anything about Optuna!")
    prompt = st.text_input(
        "Your question:",
        placeholder="e.g., How do I create a simple optimization study?",
        key="user_input"
    )

    # Handle user input
    if prompt:
        with st.spinner("🤔 Thinking..."):
            generated_response = run_llm(
                query=prompt,
                chat_history=st.session_state["chat_history"]
            )
            
            sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
            st.session_state["current_sources"] = sources
            
            formatted_response = f"{generated_response['result']}"
            
            st.session_state["chat_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", generated_response["result"]))

    # Display chat history
    if st.session_state["chat_history"]:
        st.markdown("### Conversation History")
        for generated_response, user_prompt in zip(
            st.session_state["chat_answer_history"],
            st.session_state["chat_prompt_history"]
        ):
            message(user_prompt, is_user=True, key=str(user_prompt))
            message(generated_response, key=str(generated_response))
            
        # Show sources for the latest response
        if st.session_state["current_sources"]:
            with st.expander("View Sources"):
                for source in st.session_state["current_sources"]:
                    st.markdown(f"- {source}")

with tab2:
    st.markdown("### Reference Documentation")
    st.markdown("""
    This RAG assistant is powered by the following Optuna documentation:
    - Installation Guide
    - Tutorial Documentation
    - API Reference
    - Examples and Case Studies
    """)
