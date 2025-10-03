from typing import Any
import chromadb
from chromadb.errors import NotFoundError
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
        self.name_collection = collection_name

        try:
            self.client.get_collection(name=self.name_collection)
            exists = True

        except NotFoundError:
            exists = False

        if exists:
            raise CustomError(f"Collection {collection_name} already exists")
        else:
            self.client.create_collection(name=collection_name)
    
    def DeleteChromaCollection(self, collection_name: str) -> None:
        """
        this function deletes a collection in ChromaDB.

        Args:
            collection_name (str): The name of the collection to be deleted.
        """
        try:
            self.client.get_collection(name=collection_name)
        except NotFoundError:
            raise CustomError(f"Collection {collection_name} does not exist")
        else:
            self.client.delete_collection(collection_name)

    def QueryChroma(self, text: str, results: int = 3) -> dict:
        """
        This function queries the ChromaDB collection using the provided text and returns the most similar results.

        Args:
            text (str): The input text to be queried.
            results (int): The number of similar results to return. Default is 3.
        Returns:
            dict: A dictionary containing the query results.
        """
        if not text:
            return {}

        embedder = Embedder()
        query_embedding = embedder.EmbedQuery(text)

        try:
            collection = self.client.get_collection(name=self.name_collection)
        except NotFoundError:
            raise CustomError(f"Collection '{self.name_collection}' does not exist. Please create the collection first.")
        
        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=results
        )

        docs = query_results.get("documents")
        metas = query_results.get("metadatas")
        dists = query_results.get("distances")

        top_doc = docs[0][:results] if docs else None
        top_meta = metas[0][:results] if metas else None
        top_score = dists[0][:results] if dists else None

        return {
            "document": top_doc,
            "metadata": top_meta,
            "distance": top_score
        }
