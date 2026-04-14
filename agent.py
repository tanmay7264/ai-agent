from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import load_vectorstore


def _get_llm(provider: str, model_name: str, api_key: str):
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.3)
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.3)
    raise ValueError(f"Unknown provider: {provider}")


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question using ONLY the context below.
If the answer is not found in the context, say you don't have enough information.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


class RAGAgent:
    """Conversational RAG agent using LangChain LCEL (modern approach)."""

    def __init__(self, provider: str, model_name: str, api_key: str):
        vectorstore = load_vectorstore()
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = _get_llm(provider, model_name, api_key)
        self.chat_history: list = []

    def __call__(self, inputs: dict) -> dict:
        question = inputs["question"]

        # 1. Retrieve relevant document chunks
        source_docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        # 2. Run LLM with context + history
        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context,
            "chat_history": self.chat_history,
        })

        # 3. Update conversation history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return {"answer": answer, "source_documents": source_docs}


def build_agent(provider: str, model_name: str, api_key: str) -> RAGAgent:
    return RAGAgent(provider, model_name, api_key)
