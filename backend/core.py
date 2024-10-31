# Import dotenv e carregar variáveis de ambiente
import os
from dotenv import load_dotenv
load_dotenv()

# Importar classe de retrieval chain, que pega os documentos relevantes do Pinecone
from langchain.chains.retrieval import create_retrieval_chain

# Importar hub para prompts já feitos
from langchain import hub

# Pega o prompt, coloca a docuemtação nele, e envia opara o LLM
from langchain.chains.combine_documents import create_stuff_documents_chain

# Chain ue remonta pergunta principal (qquery) quando há um hisórico de chat
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# Importar buscador de contexto pinecone
from langchain_pinecone import PineconeVectorStore

# Importar pacote de embeddings da voyage ai
from langchain_voyageai import VoyageAIEmbeddings

# Importar modelo claude
from langchain_anthropic import ChatAnthropic

# Definir função para rodar llm RAG
def run_llm(query, chat_history=[]):
    # Setar modelo de embedding
    emmbeding = VoyageAIEmbeddings(model="voyage-3")

    # Setar o objeto de busca de similaridade no Pinecone
    docseacrh = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=emmbeding)

    # Setar modelo de chat
    chat = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)

    # Fazer prompt de retrieval QA
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Combinar prompt com chat
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # Fazer prompt de repharase (para casos onde tem histórico de conversa a pergunta precisa ser re-fraseada)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docseacrh.as_retriever(), prompt=rephrase_prompt,
    )

    # Criar chain de busca -> prompt + chat (Rag Model)
    qa = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain=stuff_documents_chain,
    )

    # Invocar a chain com a query do usuário
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    # Ajustar o dicionário do result, e retornar (front end purposes) 
    return {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

# Executar como script
if __name__ == "__main__":
    # Chamar função	com query
    res = run_llm(query="What is optuna?")

    # Mostrar resposta
    print(res["answer"])
