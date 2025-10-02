from typing import Any
import chromadb
from embedding import Embedder


class CustomError(Exception):
    """
    this is a custom error class for handling specific exceptions.
    """
    pass


class ChromaDBHandler:
    def __init__(self, db_path: str = "chroma_db"):
        """
        Initializes the ChromaDBHandler with a persistent ChromaDB client.
        which makes collection in disk.

        Args:
            db_path (str): The file path where the ChromaDB database will be stored.
        """
        self.client = chromadb.PersistentClient(path=db_path)

    def CreateChromaCollection(self, collection_name: str) -> None:
        """
        this function creates a collection in ChromaDB.

        Args:
            collection_name (str): The name of the collection to be created.

        Returns:
            None
        """
        if self.client.get_collection(name=collection_name):
            raise CustomError(f"Collection '{collection_name}' already exists.")
        self.client.create_collection(name=collection_name)
    
    def DeleteChromaCollection(self, collection_name: str) -> None:
        """
        this function deletes a collection in ChromaDB.

        Args:
            collection_name (str): The name of the collection to be deleted.
        """
        if not self.client.get_collection(name=collection_name):
            raise CustomError(f"Collection {collection_name} does not exist")
        
        self.client.delete_collection(collection_name)

    def GetChromaRecord(self, collection_name: str, id: str) -> dict:
        pass

        
