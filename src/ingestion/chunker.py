"""
Document Chunker Module
Implements various chunking strategies for document processing.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    
    content: str
    metadata: Dict = field(default_factory=dict)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0
    
    def __post_init__(self):
        if self.chunk_id is None:
            id_string = f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata
        }


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str = None, metadata: Dict = None) -> List[Chunk]:
        pass
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunking - fixed-size chunks with overlap."""
    
    def chunk(self, text: str, doc_id: str = None, metadata: Dict = None) -> List[Chunk]:
        text = self._clean_text(text)
        metadata = metadata or {}
        chunks = []
        
        if len(text) <= self.chunk_size:
            return [Chunk(content=text, metadata=metadata, doc_id=doc_id, chunk_index=0)]
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            if end < len(text):
                for sep in ['. ', '! ', '? ', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size * 0.5:
                        end = start + last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(content=chunk_text, metadata=metadata, doc_id=doc_id, chunk_index=chunk_index))
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start >= len(text) - self.min_chunk_size:
                break
        
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking - splits at paragraph/sentence boundaries."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk(self, text: str, doc_id: str = None, metadata: Dict = None) -> List[Chunk]:
        text = self._clean_text(text)
        metadata = metadata or {}
        chunks = []
        
        segments = self._split_into_paragraphs(text)
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for segment in segments:
            if current_size + len(segment) > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(content=chunk_text, metadata=metadata, doc_id=doc_id, chunk_index=chunk_index))
                    chunk_index += 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(segment)
            current_size += len(segment)
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(content=chunk_text, metadata=metadata, doc_id=doc_id, chunk_index=chunk_index))
        
        return chunks


def get_chunker(strategy: str = "semantic", **kwargs) -> BaseChunker:
    """Factory function to get a chunker by strategy name."""
    chunkers = {"semantic": SemanticChunker, "sliding_window": SlidingWindowChunker}
    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}")
    return chunkers[strategy](**kwargs)
