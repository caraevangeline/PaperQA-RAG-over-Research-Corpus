"""
Document ingestion pipeline for the RAG chatbot.

Loads PDF research papers, splits them into overlapping chunks, generates
embeddings, and persists a Chroma vector store to disk.  Run this script
once (or whenever the paper collection changes) before starting the app.

Usage::

    python ingest.py
"""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAPERS_DIR = "papers"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_papers(papers_dir: str = PAPERS_DIR) -> list:
    """
    Load all PDF files from *papers_dir* into LangChain Document objects.

    Each page becomes one Document; the originating filename is stored in
    ``metadata["source"]`` so sources can be cited at query time.

    Args:
        papers_dir: Path to the directory that contains the PDF files.

    Returns:
        Flat list of Document objects (one per page across all PDFs).
    """
    docs = []
    for filename in os.listdir(papers_dir):
        if not filename.endswith(".pdf"):
            continue
        print(f"Loading: {filename}")
        loader = PyPDFLoader(os.path.join(papers_dir, filename))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        docs.extend(pages)
    print(f"Loaded {len(docs)} pages from {papers_dir}")
    return docs


def chunk_documents(
    docs: list,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """
    Split documents into overlapping text chunks suitable for retrieval.

    Uses a hierarchy of separators (paragraph → line → sentence → word) so
    splits occur at natural boundaries whenever possible.

    Args:
        docs: List of LangChain Document objects to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks to
            preserve context across boundaries.

    Returns:
        List of chunked Document objects with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vectorstore(
    chunks: list,
    persist_dir: str = VECTORSTORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """
    Embed document chunks and persist them in a Chroma vector store.

    Args:
        chunks: List of chunked Document objects to embed.
        persist_dir: Local directory where the vector store is saved.
        embedding_model: HuggingFace model name used for embedding generation.

    Returns:
        Populated and persisted Chroma instance.
    """
    print("Building vector store — this may take a few minutes...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print("Vector store built and saved.")
    return vectorstore


def load_vectorstore(
    persist_dir: str = VECTORSTORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """
    Load a previously persisted Chroma vector store from disk.

    Args:
        persist_dir: Directory where the vector store was saved.
        embedding_model: Must match the model used when the store was built.

    Returns:
        Chroma instance ready for similarity search.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


if __name__ == "__main__":
    docs = load_papers()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
