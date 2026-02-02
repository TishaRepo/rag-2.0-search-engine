"""
Pinecone Retriever
Semantic search using Pinecone vector database + Local Embeddings.
Optimized for memory-capable environments (like HF Spaces).
"""

import logging
from typing import List, Dict, Optional
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from .bm25_retriever import RetrievalResult
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class PineconeRetriever:
    """
    Pinecone Retriever using local SentenceTransformer for embedding generation.
    Ideal for environments with >2GB RAM.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION
    ):
        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.dimension = dimension
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY must be provided")

        self.pc = Pinecone(api_key=self.api_key)
        
        # Load local embedding model
        logger.info(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name} (Dim: {self.dimension})")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Pinecone local retriever initialized.")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        return self.model.encode([text])[0].tolist()

    def add_documents(self, chunks: List[Dict], batch_size: int = 32) -> int:
        """Add documents to Pinecone using local embeddings."""
        if not chunks:
            return 0
            
        ids = [c.get("chunk_id", c.get("id")) for c in chunks]
        texts = [c.get("content", c.get("text", "")) for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        
        # Inject text into metadata for retrieval
        for i, meta in enumerate(metadatas):
            meta["text"] = texts[i]
            
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
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
        """Retrieve using local embeddings + Pinecone cloud search."""
        query_embedding = self.get_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
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
