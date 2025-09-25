"""
this file is for chunking modules
multiple kind of chunkings are covered in it
"""
"""
these are libraries for chunking if needed
"""
from typing import Any# this library is used for typehints



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
        return NotImplementedError

class TokenBasedChunking(Chunking):
    """
    this class is made for token-based chunking with specific number of tokens
    """
    def __init__(self, tokens: int = 300):
        """
        Initialization of needed arguments
        """
        self.num_tokens = tokens
        self.chunks = []
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
        tokens = text.split()
        chunk_num = 1
        chunk_meta_data = meta_data.copy()

        for i in range(0, len(tokens), self.num_tokens):

            self.chunks.append({
                                "chunk_num": {chunk_num},
                                "chunk": " ".join(tokens[i: i+self.num_tokens]),
                                "meta_data": {**meta_data,
                                              "tokens": len(tokens[i: i+self.num_tokens]),
                                              "chunking_type": self.chunking_type
                                              }
                                })
            chunk_num += 1
        
        return self.chunks
    
    def __str__(self):
        """
        this method returns string representation of class
        """
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
        lines = text.split('\n')
        chunk_num = 1
        chunk_meta_data = meta_data.copy()

        if lines:
            for i in range(0, len(lines), self.lines):
                self.chunks.append({
                    "chunk_num": chunk_num,
                    "chunk": " ".join(lines[i: i+self.lines]),
                    "meta_data": {**meta_data,
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
        paragraphs = text.split('\n\n')
        chunk_num = 1
        chunk_meta_data = meta_data.copy()
        if paragraphs:
            for paragraph in paragraphs:
                self.chunks.append({
                    "chunk_num": chunk_num,
                    "chunk": paragraph,
                    "meta_data": {**meta_data,
                                    "paragraph_length": len(paragraph),
                                    "chunking_type": "ParagraphBased"
                                    }
                    })
                chunk_num += 1
            return self.chunks
        else:
            return [{"error": "no paragraphs found"}]

