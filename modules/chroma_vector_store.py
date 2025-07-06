from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List, Dict, Any
import os
from modules.config import *


class ChromaVectorStore:
    def __init__(self, embeddings_model: str = EMBEDDINGS_MODEL_NAME):
        """Initialize the ChromaVectorStore with Ollama embeddings.

        Args:
            embeddings_model: Name of the embeddings model in Ollama
        """
        self.embeddings = OllamaEmbeddings(model=embeddings_model)
        self.vector_store = None
        self.persist_directory = "chroma_db"

    def update_vector_store(self, processed_images: List[Dict[str, Any]]):
        """Create or update the vector store with new processed image data."""
        if not processed_images:
            print("No new images to add to the store.")
            return

        # Create documents for vector store
        documents = []
        for img_data in processed_images:
            content = f"""
            {img_data['description']}
            """

            doc = Document(
                page_content=content,
                metadata={
                    "image_path": img_data["image_path"],
                    "filename": img_data["filename"],
                    "file_size": img_data["file_size"],
                },
            )
            documents.append(doc)

        # If store exists, load it and add documents. Otherwise, create a new one.
        if os.path.exists(self.persist_directory) and self.vector_store is None:
            self.load_vector_store()

        if self.vector_store:
            print(f"Adding {len(documents)} new documents to the existing vector store.")
            self.vector_store.add_documents(documents=documents)
        else:
            print(f"Creating new vector store with {len(documents)} documents.")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

        print("Vector store update complete.")

    def load_vector_store(self):
        """Load existing vector store."""
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            print("Vector store loaded successfully")
        else:
            print("No existing vector store found")

    def get_existing_filenames(self) -> set:
        """Get a set of all filenames currently in the vector store."""
        if not os.path.exists(self.persist_directory):
            return set()

        # Load the store to query it, if not already loaded
        if self.vector_store is None:
            self.load_vector_store()

        if not self.vector_store:
            return set()

        # .get() retrieves all documents and metadata without performing a search
        response = self.vector_store.get()

        # Extract filenames from metadata
        existing_filenames = {
            meta["filename"] for meta in response.get("metadatas", []) if "filename" in meta
        }
        return existing_filenames

    def search_photos(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for photos using natural language query."""
        if not self.vector_store:
            print("Vector store not initialized. Please process images first.")
            return []

        # Perform similarity search
        results = self.vector_store.similarity_search_with_score(query, k=k)

        search_results = []
        for doc, score in results:
            search_results.append(
                {
                    "image_path": doc.metadata["image_path"],
                    "filename": doc.metadata["filename"],
                    "similarity_score": score,
                    "content": doc.page_content,
                }
            )

        return search_results
