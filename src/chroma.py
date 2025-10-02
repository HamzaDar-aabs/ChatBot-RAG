import chromadb
from embedding import Embedder

sample_chunks = [
    {
        "chunk_num": 1,
        "chunk": "Artificial intelligence is transforming industries.",
        "meta_data": {
            "tokens": 6,
            "chunking_type": "fixed_size",
            "source": "article1"
        }
    },
    {
        "chunk_num": 2,
        "chunk": "Chroma is a vector database for RAG applications.",
        "meta_data": {
            "tokens": 8,
            "chunking_type": "fixed_size",
            "source": "article2"
        }
    },
    {
        "chunk_num": 3,
        "chunk": "Custom embeddings can be integrated into Chroma.",
        "meta_data": {
            "tokens": 7,
            "chunking_type": "fixed_size",
            "source": "article3"
        }
    }
]

embedder = Embedder()
embdings = embedder.embed(sample_chunks)

# print(embdings)



chroma_client = chromadb.PersistentClient(path= "src/chroma_db")
collection = chroma_client.get_or_create_collection(name="my_collection")
collection.add(
    ids=[f"doc{c['chunk_num']}" for c in embdings],
    documents=[c["chunk"] for c in embdings],
    embeddings=[c["embedding"] for c in embdings],
    metadatas=[c["meta_data"] for c in embdings]
)

collection_data = collection.get(limit=1, offset=0, include=["metadatas", "documents", "embeddings"])
print(collection_data)
# chroma_client.delete_collection(name="my_collection")
