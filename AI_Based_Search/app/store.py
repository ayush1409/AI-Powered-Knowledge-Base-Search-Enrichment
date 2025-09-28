from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import FAISS_DIR, OPENAI_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
import os
from pathlib import Path
from typing import List, Tuple
import faiss

embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL, 
    openai_api_key=None if not os.getenv("OPENAI_API_KEY") else os.getenv("OPENAI_API_KEY")
)

class VectorStoreManager:
    def __init__(self, persist_directory: str = str(FAISS_DIR)):
        self.persist_directory = persist_directory
        self.index_path = Path(self.persist_directory)
        self.store = None  # placeholder until explicitly loaded or created

        # Load a fresh index everytime
        # index = FAISS.
        # index = faiss.read_index(self.index_path)
        # index.reset()

    def load_or_create_store(self, embeddings):
        """
        Loads an existing FAISS store from disk if available,
        otherwise creates a new one and saves it.
        """
        # index_file = self.index_path / "index.faiss"
        index_file = Path(os.path.join(self.index_path, 'index.faiss'))
        print(f"{index_file=}")
        if self.index_path.exists():
            # Load existing FAISS index
            # index = faiss.read_index(self.persist_directory)
            # index.reset()
            self.store = FAISS.load_local(
                self.persist_directory, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            # Create a new empty FAISS index
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.store = FAISS.from_texts([], embeddings)
            self.store.save_local(self.persist_directory)

        return self.store

    def add_documents(self, docs, embeddings = embeddings):
        """ 
        Adds new documents into the FAISS store and persists them.
        """
        if self.store is None:
            self.load_or_create_store(embeddings)

        self.store.add_documents(docs)
        self.store.save_local(self.persist_directory)

    def similarity_search_with_score(self, query, embeddings=embeddings, k=2):
        """
        Runs a similarity search against the FAISS store.
        """
        if self.store is None:
            self.load_or_create_store(embeddings)

        return self.store.similarity_search_with_score(query, k=k)