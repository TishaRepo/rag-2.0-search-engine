"""
Pinecone Retriever
Semantic search using Pinecone vector database.
"""

import logging
from typing import List, Dict, Optional
import time

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from .bm25_retriever import RetrievalResult
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class PineconeRetriever:
    """
    Pinecone-based Semantic Retriever.
    
    Uses Pinecone cloud vector database for scalable semantic search.
    Provides persistence and high availability for document retrieval.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = EMBEDDING_DIMENSION
    ):
        """
        Initialize Pinecone Retriever.
        """
        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.dimension = dimension
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY must be provided for PineconeRetriever")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists, create if not
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info(f"Pinecone retriever initialized. Index: {self.index_name}")

    def add_documents(
        self,
        chunks: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the Pinecone index.
        """
        if not chunks:
            return 0
        
        ids = [c.get("chunk_id", c.get("id")) for c in chunks]
        texts = [c.get("content", c.get("text", "")) for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        
        # Inject content into metadata for retrieval
        for i, meta in enumerate(metadatas):
            meta["text"] = texts[i]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
        
        # Upsert to Pinecone
        vectors = []
        for i in range(len(ids)):
            vectors.append({
                "id": ids[i],
                "values": embeddings[i].tolist(),
                "metadata": metadatas[i]
            })
            
        # Batch upsert
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
            
        logger.info(f"Added {len(chunks)} documents to Pinecone")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve semantically similar documents from Pinecone.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        for match in results.get("matches", []):
            score = match.get("score", 0.0)
            if score < score_threshold:
                continue
                
            metadata = match.get("metadata", {})
            content = metadata.pop("text", "")
            
            retrieval_results.append(RetrievalResult(
                chunk_id=match.get("id"),
                content=content,
                score=float(score),
                metadata=metadata,
                retrieval_method="pinecone"
            ))
            
        return retrieval_results

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a text."""
        return self.embedding_model.encode([text])[0].tolist()
