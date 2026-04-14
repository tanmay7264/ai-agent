from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rag_pipeline import load_vectorstore


def _get_llm(provider: str, model_name: str, api_key: str):
    """Return the appropriate LangChain chat model for the chosen provider."""
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.3)

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.3)

    raise ValueError(f"Unknown provider: {provider}")


def build_agent(provider: str, model_name: str, api_key: str) -> ConversationalRetrievalChain:
    """
    Build a conversational RAG chain:
      User question → FAISS retrieval → context + history → LLM → answer
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = _get_llm(provider, model_name, api_key)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain
