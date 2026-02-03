"""
Vector Retriever
Semantic search using dense vector embeddings.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
import chromadb

from .bm25_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Vector-based Semantic Retriever.
    
    Uses dense embeddings to find semantically similar documents.
    Unlike BM25, this finds related concepts even without exact word matches.
    
    How it works:
    1. Convert query to 384-dim vector using sentence-transformers
    2. Find documents with similar vectors using cosine similarity
    3. Return top-k most similar documents
    
    Best for: Conceptual queries, synonyms, paraphrased questions
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "ngse_documents"
    ):
        """
        Initialize Vector Retriever.
        
        Args:
            embedding_model: Sentence-transformer model name
            chroma_persist_dir: Path to ChromaDB persistence directory
            collection_name: Name of ChromaDB collection
        """
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        if chroma_persist_dir:
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        else:
            self.chroma_client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector retriever initialized. Collection has {self.collection.count()} documents")
    
    def add_documents(
        self,
        chunks: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            chunks: List of dicts with 'chunk_id', 'content', 'metadata'
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of documents added
        """
        if not chunks:
            return 0
        
        ids = [c.get("chunk_id", c.get("id")) for c in chunks]
        texts = [c.get("content", c.get("text", "")) for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=batch_size
        )
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} documents to vector store")
        return len(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        where: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve semantically similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            where: Optional metadata filter (ChromaDB where clause)
            
        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Query ChromaDB
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where:
            query_params["where"] = where
        
        results = self.collection.query(**query_params)
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distances; convert to similarity (1 - distance for cosine)
            distance = results['distances'][0][i]
            similarity = 1 - distance  # For cosine distance
            
            if similarity < score_threshold:
                continue
            
            retrieval_results.append(RetrievalResult(
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i] if results['documents'] else "",
                score=float(similarity),
                metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                retrieval_method="vector"
            ))
        
        logger.debug(f"Vector retrieved {len(retrieval_results)} results for query: {query[:50]}...")
        return retrieval_results
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a text."""
        return self.embedding_model.encode([text])[0].tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        import numpy as np
        
        emb1 = self.embedding_model.encode([text1])[0]
        emb2 = self.embedding_model.encode([text2])[0]
        
        # Cosine similarity
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)
    
    def explain_retrieval(self, query: str, chunk_id: str) -> Dict:
        """
        Explain why a document was retrieved.
        Shows similarity score and embedding details.
        """
        # Get the chunk
        chunk_data = self.collection.get(ids=[chunk_id], include=["documents", "embeddings"])
        
        if not chunk_data['ids']:
            return {"error": "Chunk not found"}
        
        content = chunk_data['documents'][0] if chunk_data['documents'] else ""
        
        # Calculate similarity
        similarity = self.similarity(query, content)
        
        return {
            "chunk_id": chunk_id,
            "similarity_score": similarity,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": 384,
            "explanation": f"Semantic similarity: {similarity:.3f} (1.0 = identical meaning)"
        }


if __name__ == "__main__":
    # Test Vector Retriever
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import INDICES_DIR
    
    # Load existing index
    retriever = VectorRetriever(
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        collection_name="ngse_documents"
    )
    
    # Test queries
    queries = [
        "What is AI?",  # Should match "artificial intelligence"
        "How do computers learn?",  # Should match "machine learning"
        "Where is the server?"  # Should match asset document
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score: {r.score:.3f}")
            print(f"      {r.content[:80]}...")
    
    # Test semantic similarity
    print("\n📊 Semantic Similarity Examples:")
    pairs = [
        ("machine learning", "ML"),
        ("artificial intelligence", "AI"),
        ("pizza", "machine learning")
    ]
    for t1, t2 in pairs:
        sim = retriever.similarity(t1, t2)
        print(f"  '{t1}' ↔ '{t2}': {sim:.3f}")
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        delete_all: bool = False
    ) -> int:
        """
        Delete documents from ChromaDB.
        """
        if delete_all:
            count = self.collection.count()
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared ChromaDB collection ({count} docs)")
            return count
            
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} docs from ChromaDB by ID")
            return len(ids)
            
        if where:
            # Note: ChromaDB delete doesn't return count directly
            self.collection.delete(where=where)
            logger.info(f"Deleted docs from ChromaDB using where filter: {where}")
            return 1
            
        return 0
