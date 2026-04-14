from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index"
DATA_DIR = "data"


def load_documents():
    """Load all PDFs and .txt files from the data/ directory."""
    docs = []
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)

    for pdf_file in data_path.glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    for txt_file in data_path.glob("**/*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs.extend(loader.load())

    return docs


def get_embeddings():
    """Return free, local HuggingFace sentence embeddings."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(docs):
    """Chunk documents, embed them, and save a FAISS index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore, len(chunks)


def load_vectorstore():
    """Load an existing FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def vectorstore_exists():
    return Path(FAISS_INDEX_PATH).exists()
