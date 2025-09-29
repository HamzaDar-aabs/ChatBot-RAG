from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model: str = 'all-MiniLM-L6-v2'):
        """
        initialization of parameters needed
        """
        self.model = SentenceTransformer(model)

    def embed(self, input: list[dict]) -> list[dict]:
        """
        This function takes a text input and returns its embedding using the specified model.

        Args:
            input (list[dict]): A list of dictionaries, having chunks of text to be embedded. Each dictionary should have a 'chunk' key.

        returns:
            list[dict]: A list of dictionaries with the original text and its corresponding embedding.
        """
        self.input = input

        if self.input:
            # this is batch embedding for all the chunks
            all_chunks = [chunk["chunk"] for chunk in self.input] # collecting all chunks at once
            embeddings = self.model.encode(all_chunks, convert_to_tensor=True) # encoding all chunks at once

            for i, embedding in zip(self.input, embeddings): # looping trough each chunk and its corresponding embedding
                i["embedding"] = embedding.cpu().numpy() # adding a list of embeddings to each chunk


            return self.input
        else:
            return [{'error': 'No text provided for embedding.'}]
        
test_input = [
    {"chunk": "Cats are playful animals.", "metadata": {"source": "doc1", "page": 1}},
    {"chunk": "Dogs are loyal companions.", "metadata": {"source": "doc1", "page": 2}},
    {"chunk": "Birds can fly high in the sky.", "metadata": {"source": "doc2", "page": 5}},
]

embedder = Embedder()
embeddings = embedder.embed(test_input)
print(test_input,"\n\n",embeddings)
