import os
from pathlib import Path
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma



def load_documents(directory: str) -> List[str]:
    """
    Load documents from a directory, admits .txt, .md, and .pdf files.

    :param directory: Path to the directory containing the documents.
    :return: List of documents.
    """
    input_path = Path(directory)
    if not input_path.exists():
        raise ValueError(f"Input path {input_path} does not exist.")

    txt_loader = DirectoryLoader(directory,
                                 glob="**/*.txt",
                                 loader_cls=TextLoader,
                                 show_progress=True,
                                 use_multithreading=True)
    text_docs = txt_loader.load()

    md_loader = DirectoryLoader(directory,
                                glob="**/*.md",
                                loader_cls=TextLoader,
                                show_progress=True,
                                use_multithreading=True)
    md_docs = md_loader.load()

    pdf_loader = DirectoryLoader(directory,
                                 glob="**/*.pdf",
                                 loader_cls=PyPDFLoader,
                                 show_progress=True,
                                 use_multithreading=True)
    pdf_docs = pdf_loader.load()

    return text_docs + md_docs + pdf_docs

def split_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len,
                                                   is_separator_regex=True,
                                                   separators=["\n\n", "\n", ".", "!", "?", " ", "#", "##", "###"])
    split_documents = text_splitter.split_documents(documents)

    return split_documents

def build_vector_store(docs: List,
                       persist_dir: str,
                       collection_name: str,
                       embedding_model: str = "granite-embedding:278m") -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma.from_documents(documents=docs,
                                         embedding=embeddings,
                                         persist_directory=persist_dir,
                                         collection_name=collection_name)

    return vector_store

def get_vector_store(persist_dir: str,
                     collection_name: str,
                     embedding_model: str = "granite-embedding:278m") -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(persist_directory=persist_dir,
                          collection_name=collection_name,
                          embedding_function=embeddings)

    return vector_store


def query_vector_store(vector_store: Chroma, query: str, k: int = 5) -> List[dict]:
    results = vector_store.similarity_search_with_score(query, k=k)

    return results


def main():
    input_docs_dir = os.environ.get("RAG_INPUT_DIR", "data")
    persist_dir = os.environ.get("RAG_CHROMA_DIR", "chroma_db")
    collection_name = os.environ.get("RAG_COLLECTION", "docs")
    embedding_model = os.environ.get("RAG_EMBED_MODEL", "granite-embedding:278m")

    print(f"Loading raw documents from: {input_docs_dir}")
    raw_docs = load_documents(input_docs_dir)
    print(f"Loaded {len(raw_docs)} raw documents")

    print("Splitting documents into chunks...")
    chunks = split_documents(raw_docs, chunk_size=1000, chunk_overlap=150)
    print(f"Produced {len(chunks)} chunks")

    print("Building and persisting Chroma vector store...")
    vs = build_vector_store(
        docs=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    print("Chroma vector store persisted.")


if __name__ == "__main__":
    main()
