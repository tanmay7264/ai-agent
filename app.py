import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    DATA_DIR,
    build_vectorstore,
    load_documents,
    vectorstore_exists,
)
from agent import build_agent

load_dotenv()

def get_secret(key: str) -> str:
    """Read from Streamlit Cloud secrets first, then fall back to .env."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

# ── Provider / model catalogue ────────────────────────────────────────────────
PROVIDERS = {
    "Groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "key_env": "GROQ_API_KEY",
        "signup_url": "https://console.groq.com",
        "note": "Free tier — no credit card required.",
    },
    "Gemini": {
        "models": [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
        ],
        "key_env": "GOOGLE_API_KEY",
        "signup_url": "https://aistudio.google.com/app/apikey",
        "note": "Free tier via Google AI Studio.",
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 End-to-End AI Agent")
st.caption("RAG • FAISS • LangChain — Free Online LLMs (Groq / Gemini)")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 LLM Provider")

    provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    info = PROVIDERS[provider]

    model_name = st.selectbox("Model", info["models"])

    # Pre-fill from .env if available, otherwise let user type
    default_key = get_secret(info["key_env"])
    api_key = st.text_input(
        f"{provider} API Key",
        value=default_key,
        type="password",
        placeholder=f"Paste your {provider} key here",
    )
    st.caption(f"{info['note']} Get one at [{info['signup_url']}]({info['signup_url']})")
    if api_key:
        st.success("✅ API key loaded")
    else:
        st.warning("⚠️ No API key found — enter it above or add to Streamlit secrets")

    st.divider()
    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or Text Files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        Path(DATA_DIR).mkdir(exist_ok=True)
        for file in uploaded_files:
            (Path(DATA_DIR) / file.name).write_bytes(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to data/")

    st.divider()

    if st.button("🔨 Build / Rebuild Index", type="primary", use_container_width=True):
        with st.spinner("Embedding documents — first run downloads the embedding model…"):
            docs = load_documents()
            if not docs:
                st.error("No documents found. Upload files above first.")
            else:
                _, n_chunks = build_vectorstore(docs)
                st.success(f"Indexed {len(docs)} doc(s) → {n_chunks} chunks")
                st.session_state.agent = None  # will be rebuilt on first query
                st.session_state.messages = []

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None

# ── Rebuild agent when provider/model/key changes ─────────────────────────────
agent_key = (provider, model_name, api_key)
if st.session_state.get("agent_key") != agent_key:
    st.session_state.agent = None
    st.session_state.agent_key = agent_key

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Source chunks used"):
                for src in msg["sources"]:
                    label = Path(src["source"]).name
                    page = src.get("page", "")
                    st.caption(f"**{label}**" + (f" — page {page}" if page != "" else ""))
                    st.text(src["content"][:300] + ("…" if len(src["content"]) > 300 else ""))

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents…"):
    if not api_key:
        st.error(f"Please enter your {provider} API key in the sidebar.")
    elif not vectorstore_exists():
        st.error("No index found. Upload documents and click **Build / Rebuild Index**.")
    else:
        # Lazy-build agent (only when needed, so key changes take effect)
        if st.session_state.get("agent") is None:
            with st.spinner("Connecting to LLM…"):
                st.session_state.agent = build_agent(provider, model_name, api_key)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    result = st.session_state.agent({"question": prompt})
            except Exception as e:
                st.error(f"❌ LLM Error: {str(e)}")
                st.stop()

            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", ""),
                    "content": doc.page_content,
                }
                for doc in source_docs
            ]

            st.markdown(answer)
            if sources:
                with st.expander("📄 Source chunks used"):
                    for src in sources:
                        label = Path(src["source"]).name
                        page = src.get("page", "")
                        st.caption(f"**{label}**" + (f" — page {page}" if page != "" else ""))
                        st.text(src["content"][:300] + ("…" if len(src["content"]) > 300 else ""))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    if vectorstore_exists():
        st.info("Index ready. Enter your API key and ask a question.")
    else:
        st.info("Upload documents in the sidebar and click **Build / Rebuild Index** to get started.")
