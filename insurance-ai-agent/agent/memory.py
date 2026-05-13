from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Use a local embedding model (lightweight)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_vectorstore():
    """Initializes and returns the ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    persist_dir = "./chroma_db"
    
    return Chroma(
        collection_name="insurance_agent_memory",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

def add_to_memory(text: str, metadata: dict = None):
    """Adds a new entry to the agent's long-term memory."""
    db = get_vectorstore()
    db.add_texts([text], metadatas=[metadata] if metadata else None)
    db.persist()

def query_memory(query: str):
    """Retrieves relevant context from the agent's memory."""
    db = get_vectorstore()
    docs = db.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])
