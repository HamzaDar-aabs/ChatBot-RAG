"""
this file is for chunking modules
multiple kind of chunkings are covered in it
"""
"""
these are libraries for chunking if needed
"""
from typing import Any # this library is used for typehints
from sentence_transformers import SentenceTransformer, util # this library is used for semantic chunking
import nltk
from nltk.tokenize import sent_tokenize


class Chunking:
    """
    this class provides a base for all methods to impliment
    """

    def chunk(self, text: str, meta_data: dict[str, Any]):
        """
        this method chunks text

        Args:
            text: it is text given to it
            meta_data: it is the information about the chunk

        returns:
            list of dictionaries
        """
        raise NotImplementedError

class TokenBasedChunking(Chunking):
    """
    this class is made for token-based chunking with specific number of tokens
    """
    def __init__(self, tokens: int = 300):
        """
        Initialization of needed arguments
        """

        self.chunks = []
        self.num_tokens = tokens
        self.chunking_type = 'TokenBased'

    def chunk(self, text: str, meta_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        this is the implementation of chunking
        
        Args:
            text: it is given text
            meta_data: it is data about the chunk
        
        return:
            returns list of dictionary
            """
        
        self.chunks = []
        tokens = [token.strip() for token in text.split()] # splitting text into tokens using list comprehension
        chunk_num = 1

        for i in range(0, len(tokens), self.num_tokens):

            self.chunks.append({
                                "chunk_num": chunk_num,
                                "chunk": " ".join(tokens[i: i+self.num_tokens]), # making chunk by joining tokens
                                "meta_data": {**meta_data, # this is method of unpacking dictionary
                                              "tokens": len(tokens[i: i+self.num_tokens]), # calculating number of tokens in chunk
                                              "chunking_type": self.chunking_type
                                              }
                                })
            chunk_num += 1
        
        return self.chunks

    
class LineBasedChunking(Chunking):
    """
    this class provides chunking on the basis of line instead of tokens
    """
    def __init__(self, lines: int = 30):
        """
        initialization of parameters needed
        
        Args:
            lines: how many lines per chunk
            
        returns:
            list of dictionary
        """

        self.lines = lines
        self.chunks = []
    
    def chunk(self, text: str, meta_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        this is implementation of line based chunking
        
        Args:
            text: takes text to chunk
            meta_data: information about text and chunk
        
        returns:
            list of dictionary
        """

        self.chunks = []
        lines = [line.strip() for line in text.split('\n') if line.strip()] # splitting text into lines and removing empty lines using list comprehension
        chunk_num = 1

        if lines:
            for i in range(0, len(lines), self.lines):
                self.chunks.append({
                    "chunk_num": chunk_num,
                    "chunk": " ".join(lines[i: i+self.lines]),
                    "meta_data": {**meta_data, # unpacking dictionary
                                    "lines": len(lines[i: i+self.lines]),
                                    "chunking_type": "LineBased"
                                    }
                    })
                chunk_num += 1
            return self.chunks
        else:
            return [{"error": "no lines found"}]

    

class ParagraphBasedChunking(Chunking):
    """
    this class provides chunking on the basis of paragraph instead of tokens or lines
    """
    def __init__(self):
        """
        initialization of parameters needed
        """
        self.chunks = []
    
    def chunk(self, text: str, meta_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        this is implementation of paragraph based chunking
        
        Args:
            text: takes text to chunk
            meta_data: information about text and chunk
        
        returns:
            list of dictionary
        """

        self.chunks = []
        paragraphs = [paragraph.strip() for paragraph in text.split('\n\n') if paragraph.strip()]
        chunk_num = 1

        if paragraphs:
            for paragraph in paragraphs:
                self.chunks.append({
                    "chunk_num": chunk_num,
                    "chunk": paragraph,
                    "meta_data": {**meta_data, # unpacking dictionary
                                    "paragraph_length": len(paragraph),
                                    "chunking_type": "ParagraphBased"
                                    }
                    })
                chunk_num += 1
            return self.chunks
        else:
            return [{"error": "no paragraphs found"}]

class SlidingWindowChunking(Chunking):
    """
    this class provides chunking on the basis of sliding window
    """
    def __init__(self, window_size: int = 500, over_lap: int = 100):
        """
        initialization of parameters needed
        
        Args:
            window_size: size of window
            step_size: step size for sliding window
            
        returns:
            list of dictionary
        """
        self.window_size = window_size
        self.over_lap = over_lap
        self.chunks = []
    
    def chunk(self, text: str, meta_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        this is implementation of sliding window based chunking
        
        Args:
            text: takes text to chunk
            meta_data: information about text and chunk
        
        returns:
            list of dictionary
        """
        self.chunks = []
        tokens = text.split()
        chunk_num = 1
        start = 0

        if tokens:
            while start < len(tokens):
                self.chunks.append({
                    "chunk_num": chunk_num,
                    "chunk": " ".join(tokens[start: start+self.window_size]),
                    "meta_data": {**meta_data, # unpacking dictionary
                                    "tokens": len(tokens[start: start+self.window_size]),
                                    "chunking_type": "SlidingWindow"
                                    }
                    })
                chunk_num += 1
                start += self.window_size - self.over_lap # move back by overlap amount
        
            return self.chunks
        else:
            return [{"error": "no tokens found"}]
        
class AdjacentSemanticChunking(Chunking): # based on lines
    """
    this class provides chunking on the basis of semantic similarity
    """
    def __init__(self, model: Any = 'all-MiniLM-L6-v2', threshold: float = 0.15):
        """
        initialization of parameters needed
        
        Args:
            model: it is a model used for making chunks based on semantic similarity
            
        returns:
            list of dictionary
        """

        # here the model tokenizer is downloaded if not present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

        self.model = SentenceTransformer(model)
        self.threshold = threshold
        self.chunks = []
    
    def chunk(self, text: str, meta_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        this is implementation of semantic based chunking
        
        Args:
            text: takes text to chunk
            meta_data: information about text and chunk
        
        returns:
            list of dictionary
        """

        self.chunks = []
        sentences = sent_tokenize(text)
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        chunk_num = 1
        sent_num =  0

        if sentences:
            # here we are calculating cosine similarity between consecutive sentences
            similarities = util.cos_sim(embeddings[:-1], embeddings[1:]).diagonal() # diagonal method is used to get the diagonal elements of the similarity matrix
            """Matrix (3x3):
                [[sim(s0,s1), sim(s0,s2), sim(s0,s3)]
                [sim(s1,s1), sim(s1,s2), sim(s1,s3)]
                [sim(s2,s1), sim(s2,s2), sim(s2,s3)]]

                Diagonal: [sim(s0,s1), sim(s1,s2), sim(s2,s3)]
            """
            current_chunk_sent = [embeddings[sent_num]]

            for i, sim in enumerate(similarities):
                if sim >= self.threshold:
                    current_chunk_sent.append(embeddings[i+1])
                else:
                    chunk = " ".join(sentences[sent_num: i+1])
                    self.chunks.append({
                        "chunk_num": chunk_num,
                        "chunk": chunk,
                        "meta_data": {**meta_data,
                                      "sentence_length": len(sentences[sent_num: i+1]),
                                      "chunking_type": "Semantic"}
                    })
                    chunk_num += 1
                    sent_num = i + 1
                    current_chunk_sent = [embeddings[sent_num]] if sent_num < len(sentences) else []

            if sent_num < len(sentences):
                chunk = " ".join(sentences[sent_num:])
                self.chunks.append({
                    "chunk_num": chunk_num,
                    "chunk": chunk,
                    "meta_data": {**meta_data,
                                  "sentence_length": len(sentences[sent_num:]),
                                  "chunking_type": "Semantic"}
                })

            return self.chunks
        else:
            return [{"error": "no sentences found"}]