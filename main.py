# Importar classe de RAG com llm para optuna
from backend.core import run_llm

# Importar pacotes do streamlit
import streamlit as st
from streamlit_chat import message

# Mostrar o header do app
st.header("Optuna RAG Chatbot!")

# Pegar a query do usuário
prompt = st.text_input("Enter your question about Optuna!")

# Se for a primeira rodada dessa sessão streamlit
if (
    "chat_prompt_history" not in st.session_state
    and "chat_answer_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    # Define variáveis de sessão como vazias
    st.session_state["chat_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []

# Ao receber um prompt do usuário
if prompt:
    # Mostre o timeout até a resposta
    with st.spinner("Generating response..."):

        # Exeute a função RAG para pegar a resposta
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])

        # Pegue as fontes da resposta
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        # Formate a resposta antes de mostrar no streamlit
        formatted_response = f"{generated_response['result']} \n \n Sources: {'*'.join(sources)}"

        # Popule as variáveis de sessão com a iteração atual entre usuário e bot
        st.session_state["chat_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# Se a sessão atual possuir historico
if st.session_state["chat_history"]:
    # Itere entre os itens do historico
    for generated_response, user_prompt in zip(st.session_state["chat_answer_history"], st.session_state["chat_prompt_history"]):
        # Mostre eles na UI
        message(user_prompt, is_user=True)
        message(generated_response)
