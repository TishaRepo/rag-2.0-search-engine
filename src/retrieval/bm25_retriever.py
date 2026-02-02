"""
BM25 Retriever
Keyword-based retrieval using BM25 algorithm.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    retrieval_method: str = "bm25"
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "retrieval_method": self.retrieval_method
        }


class BM25Retriever:
    """
    BM25 (Best Matching 25) Retriever.
    
    BM25 is a keyword-based ranking function that scores documents based on:
    - Term frequency (TF): How often the query term appears in the document
    - Inverse document frequency (IDF): How rare the term is across all documents
    - Document length normalization: Adjusts for document size
    
    Formula: score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (docLen / avgDocLen)))
    
    Best for: Exact keyword matching, technical terms, names, IDs
    """
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize BM25 Retriever.
        
        Args:
            index_path: Path to saved BM25 index (.pkl file)
        """
        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
        self.chunk_ids: List[str] = []
        self.documents: Dict[str, Dict] = {}  # chunk_id -> {content, metadata}
        
        if index_path:
            self.load_index(index_path)
    
    def build_index(
        self, 
        chunks: List[Dict],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of dicts with 'chunk_id', 'content', 'metadata'
            save_path: Optional path to save the index
        """
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        self.corpus = []
        self.chunk_ids = []
        self.documents = {}
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", chunk.get("id"))
            content = chunk.get("content", chunk.get("text", ""))
            metadata = chunk.get("metadata", {})
            
            # Tokenize (simple whitespace tokenization)
            tokens = self._tokenize(content)
            
            self.corpus.append(tokens)
            self.chunk_ids.append(chunk_id)
            self.documents[chunk_id] = {
                "content": content,
                "metadata": metadata
            }
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(self.corpus)
        
        if save_path:
            self.save_index(save_path)
        
        logger.info(f"BM25 index built with {len(self.corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on whitespace."""
        # Remove punctuation and lowercase
        text = text.lower()
        # Keep alphanumeric and spaces
        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        return cleaned.split()
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum score to include (0.0 = include all)
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.bm25_index:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]
        
        # Build results
        results = []
        for idx, score in top_indices:
            if score < score_threshold:
                continue
                
            chunk_id = self.chunk_ids[idx]
            doc_data = self.documents.get(chunk_id, {})
            
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=doc_data.get("content", ""),
                score=float(score),
                metadata=doc_data.get("metadata", {}),
                retrieval_method="bm25"
            ))
        
        logger.debug(f"BM25 retrieved {len(results)} results for query: {query[:50]}...")
        return results
    
    def save_index(self, path: Path) -> None:
        """Save BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "corpus": self.corpus,
            "chunk_ids": self.chunk_ids,
            "documents": self.documents
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load BM25 index from disk."""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"⚠️ BM25 index not found at {path}. Knowledge base will be empty until data is ingested.")
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load BM25 index: {e}")
            return
        
        self.corpus = data.get("corpus", [])
        self.chunk_ids = data.get("chunk_ids", [])
        self.documents = data.get("documents", {})
        
        # Rebuild BM25 index
        if self.corpus:
            self.bm25_index = BM25Okapi(self.corpus)
        
        logger.info(f"BM25 index loaded: {len(self.corpus)} documents")
    
    def explain_score(self, query: str, chunk_id: str) -> Dict:
        """
        Explain why a document got its score.
        Useful for debugging and understanding results.
        """
        if chunk_id not in self.chunk_ids:
            return {"error": "Chunk not found"}
        
        idx = self.chunk_ids.index(chunk_id)
        doc_tokens = self.corpus[idx]
        query_tokens = self._tokenize(query)
        
        # Find matching terms
        matching_terms = set(query_tokens) & set(doc_tokens)
        
        return {
            "chunk_id": chunk_id,
            "query_tokens": query_tokens,
            "matching_terms": list(matching_terms),
            "match_count": len(matching_terms),
            "doc_length": len(doc_tokens),
            "explanation": f"Matched {len(matching_terms)}/{len(query_tokens)} query terms"
        }


if __name__ == "__main__":
    # Test BM25 Retriever
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import INDICES_DIR
    
    # Load existing index
    retriever = BM25Retriever(index_path=INDICES_DIR / "bm25_index.pkl")
    
    # Test queries
    queries = [
        "artificial intelligence",
        "machine learning algorithm",
        "PRIMERGY server location"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score: {r.score:.3f}")
            print(f"      {r.content[:80]}...")
