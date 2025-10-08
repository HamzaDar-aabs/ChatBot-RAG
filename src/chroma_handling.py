from typing import Any
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
import uuid


class CustomError(Exception):
    """Custom error class for ChromaDB handler."""
    pass


class ChromaDBHandler:
    def __init__(self, db_path: str = "chroma_db"):
        """
        Initializes the ChromaDBHandler with a persistent ChromaDB client.
        This stores collections on disk.
        """
        self.client = PersistentClient(path=db_path)
        self.collection = None
        self.collection_name = None

    def CreateChromaCollection(self, collection_name: str):
        """
        Creates a new collection in ChromaDB.

        Args:
            collection_name (str): The name of the collection to be created.

        Returns:
            The created collection object.
        """
        try:
            self.collection = self.client.create_collection(name=collection_name)
            self.collection_name = collection_name
        except Exception as e:
            raise CustomError(f"Collection '{collection_name}' already exists or could not be created.") from e
        
        return self.collection

    def DeleteChromaCollection(self, collection_name: str) -> None:
        """
        Deletes a collection from ChromaDB.

        Args:
            collection_name (str): The name of the collection to be deleted.
        """
        try:
            self.client.delete_collection(name=collection_name)
        except NotFoundError as e:
            raise CustomError(f"Collection '{collection_name}' does not exist.") from e
        
    
    def GetChromaCollection(self, collection_name: str) -> object:
        """
        Retrieves a collection from ChromaDB.

        Args:
            collection_name (str): The name of the collection to be retrieved.

        Returns:
            The retrieved collection object.
        """
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.collection_name = collection_name
        except NotFoundError as e:
            raise CustomError(f"Collection '{collection_name}' does not exist.") from e
        
        return self.collection


    def AddToCollection(self, input: list[dict[str, Any]], collection_name: str) -> None:
        """
        This function adds documents to the specified ChromaDB collection.
        Args:
            input (list[dict]): A list of dictionaries, each containing 'id', 'chunk', and optional 'metadata'.
            collection_name (str): The name of the collection to which the documents will be added.
        """
        self.input = input
        self.collection_name = collection_name

        if not self.input:
            raise CustomError("Input list is empty. Please provide valid data to add.")

        try:
            collection = self.client.get_collection(name=collection_name)
        except NotFoundError:
            raise CustomError(f"Collection '{collection_name}' does not exist. Please create the collection first.")

        chunks = [i['chunk'] if i['chunk'] else "NA" for i in self.input]
        metadatas = [i['meta_data'] for i in self.input] 
        ids = [str(uuid.uuid4()) for i in self.input]
        embeddings = [i['embedding'] for i in self.input]

        collection.add(
            documents=chunks, # it needs a list of strings
            metadatas=metadatas, # it needs a list of dictionaries
            ids=ids, # it needs a list of strings
            embeddings=embeddings # it needs a list of lists (2D list)
        )


    def QueryChroma(self, embedded_query: list[float], collection_name: str, results: int = 3) -> dict:
        """
        This function queries the ChromaDB collection using the provided text and returns the most similar results.

        Args:
            embedded_query (list[float]): The embedding of the query text.
            collection_name (str): The name of the collection to query.
            results (int): The number of similar results to return. Default is 3.
        Returns:
            dict: A dictionary containing the top documents, their metadata, and distances.
        """

        if not embedded_query:
            return {}

        try:
            collection = self.client.get_collection(name=collection_name)
        except NotFoundError:
            raise CustomError(f"Collection '{collection_name}' does not exist. Please create the collection first.")
        
        query_results = collection.query(
            query_embeddings=[embedded_query],
            n_results=results
        )

        # docs = query_results.get("documents", []) # getting documents from the query results
        # metas = query_results.get("metadatas", []) # getting metadatas from the query results
        # dists = query_results.get("distances", []) # getting distances from the query results

        # top_doc = docs[0][:results] if docs and docs[0] else [] # checking if docs and docs[0] are not empty
        # top_meta = metas[0][:results] if metas and metas[0] else [] # checking if metas and metas[0] are not empty
        # top_score = dists[0][:results] if dists and dists[0] else [] # checking if dists and dists[0] are not empty

        return query_results.__dict__ # this __dict__ method returns the internal dictionary of the QueryResult object
        # becuse every class object save data in the form of dictionary internally

