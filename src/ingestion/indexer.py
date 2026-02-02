"""
Document Indexer Module
Handles vector and BM25 indexing of document chunks.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import json

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Indexes document chunks for hybrid retrieval.
    Supports both vector (ChromaDB) and keyword (BM25) indexing.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_persist_dir: str = None,
        collection_name: str = "ngse_documents"
    ):
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.chroma_persist_dir = chroma_persist_dir
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        if chroma_persist_dir:
            Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        else:
            self.chroma_client = chromadb.Client()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index storage
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_chunk_ids = []
    
    def index_chunks(self, chunks: List, batch_size: int = 100) -> Dict:
        """
        Index chunks in both vector store and BM25.
        
        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for embedding
            
        Returns:
            Statistics about indexing
        """
        if not chunks:
            return {"indexed": 0}
        
        # Prepare data
        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=batch_size)
        
        # Add to ChromaDB
        logger.info("Adding to vector store...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_corpus = tokenized_corpus
        self.bm25_chunk_ids = ids
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        stats = {
            "indexed": len(chunks),
            "vector_count": self.collection.count(),
            "bm25_docs": len(self.bm25_corpus)
        }
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def save_bm25_index(self, path: Path, chunks: List = None):
        """Save BM25 index to disk with document content."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build documents dict for retriever compatibility
        documents = {}
        if chunks:
            for chunk in chunks:
                chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else chunk.get('chunk_id')
                content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else chunk.get('metadata', {})
                documents[chunk_id] = {"content": content, "metadata": metadata}
        
        data = {
            "corpus": self.bm25_corpus,
            "chunk_ids": self.bm25_chunk_ids,
            "documents": documents  # Add documents for retriever
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load_bm25_index(self, path: Path):
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25_corpus = data["corpus"]
        self.bm25_chunk_ids = data["chunk_ids"]
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"BM25 index loaded: {len(self.bm25_corpus)} documents")
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using vector similarity."""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        for i in range(len(results['ids'][0])):
            output.append({
                "chunk_id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return output
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using BM25 keyword matching."""
        if not self.bm25_index:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Get documents from ChromaDB
        chunk_ids = [self.bm25_chunk_ids[i] for i in top_indices]
        results = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        
        output = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                output.append({
                    "chunk_id": chunk_ids[i],
                    "content": results['documents'][i] if results['documents'] else "",
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "score": scores[idx]
                })
        
        return output
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "vector_count": self.collection.count(),
            "bm25_count": len(self.bm25_corpus),
            "embedding_model": self.embedding_model_name
        }


if __name__ == "__main__":
    # Test the indexer
    from document_loader import DocumentLoader
    from chunker import SemanticChunker
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import RAW_DATA_DIR, INDICES_DIR
    
    # Load sample documents
    loader = DocumentLoader()
    docs = loader.create_sample_corpus(RAW_DATA_DIR)
    
    # Chunk documents
    chunker = SemanticChunker(chunk_size=300)
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.content, doc_id=doc.doc_id, metadata=doc.metadata)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
    
    # Index chunks
    indexer = DocumentIndexer(chroma_persist_dir=str(INDICES_DIR / "chroma"))
    stats = indexer.index_chunks(all_chunks)
    print(f"Indexing stats: {stats}")
    
    # Save BM25 index
    indexer.save_bm25_index(INDICES_DIR / "bm25_index.pkl")
    
    # Test search
    query = "What is machine learning?"
    print(f"\nSearching for: '{query}'")
    
    vector_results = indexer.vector_search(query, top_k=3)
    print("\nVector Search Results:")
    for r in vector_results:
        print(f"  Score: {r['score']:.3f} - {r['content'][:100]}...")
