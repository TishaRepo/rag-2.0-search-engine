"""
Data Ingestion Module
Handles document loading, chunking, and indexing.
"""

from .document_loader import DocumentLoader
from .chunker import SemanticChunker, SlidingWindowChunker
from .indexer import DocumentIndexer

__all__ = [
    "DocumentLoader",
    "SemanticChunker", 
    "SlidingWindowChunker",
    "DocumentIndexer"
]
