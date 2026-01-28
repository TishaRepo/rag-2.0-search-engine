"""
Hybrid Retriever
Combines BM25 and Vector retrieval for best results.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass

from .bm25_retriever import BM25Retriever, RetrievalResult
from .vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class HybridResult(RetrievalResult):
    """Extended result with hybrid scoring details."""
    bm25_score: float = 0.0
    vector_score: float = 0.0
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score
        })
        return base


class HybridRetriever:
    """
    Hybrid Retriever combining BM25 and Vector search.
    
    Why hybrid?
    - BM25 excels at: Exact matches, technical terms, IDs, proper nouns
    - Vector excels at: Semantic similarity, synonyms, paraphrases
    - Hybrid combines both strengths!
    
    Fusion Methods:
    1. Weighted Sum: score = α * bm25 + (1-α) * vector
    2. Reciprocal Rank Fusion (RRF): Combines rankings, not scores
    3. Max Score: Takes the best score from either method
    
    Example:
        Query: "ML algorithms"
        - BM25 finds: Documents with "ML" and "algorithms" words
        - Vector finds: Documents about "machine learning methods"
        - Hybrid: Returns both, ranked by combined score
    """
    
    def __init__(
        self,
        bm25_index_path: Optional[Path] = None,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "ngse_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 0.5
    ):
        """
        Initialize Hybrid Retriever.
        
        Args:
            bm25_index_path: Path to BM25 index file
            chroma_persist_dir: Path to ChromaDB directory
            collection_name: ChromaDB collection name
            embedding_model: Sentence-transformer model
            alpha: Weight for BM25 (0-1). Higher = more keyword weight
        """
        self.alpha = alpha
        
        # Initialize BM25 Retriever
        self.bm25_retriever = BM25Retriever(index_path=bm25_index_path)
        
        # Initialize Vector Retriever
        self.vector_retriever = VectorRetriever(
            embedding_model=embedding_model,
            chroma_persist_dir=chroma_persist_dir,
            collection_name=collection_name
        )
        
        logger.info(f"Hybrid retriever initialized with α={alpha}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: Literal["weighted", "rrf", "max"] = "weighted",
        alpha: Optional[float] = None
    ) -> List[HybridResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Fusion method ("weighted", "rrf", "max")
            alpha: Override default alpha (optional)
            
        Returns:
            List of HybridResult objects
        """
        alpha = alpha if alpha is not None else self.alpha
        
        # Get results from both retrievers
        # Retrieve more than top_k to allow for deduplication and re-ranking
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        
        # Fuse results
        if method == "weighted":
            fused = self._weighted_fusion(bm25_results, vector_results, alpha)
        elif method == "rrf":
            fused = self._rrf_fusion(bm25_results, vector_results)
        elif method == "max":
            fused = self._max_fusion(bm25_results, vector_results)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Sort by score and return top_k
        fused.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Hybrid retrieval ({method}): {len(fused[:top_k])} results for '{query[:50]}...'")
        return fused[:top_k]
    
    def _weighted_fusion(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        alpha: float
    ) -> List[HybridResult]:
        """
        Weighted sum fusion.
        
        Formula: final_score = α * normalized_bm25 + (1-α) * vector_score
        
        Note: BM25 scores are unbounded, so we normalize them to 0-1
        """
        # Collect all unique chunks
        all_chunks: Dict[str, HybridResult] = {}
        
        # Normalize BM25 scores to 0-1 range
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results) or 1.0
        else:
            max_bm25 = 1.0
        
        # Process BM25 results
        for r in bm25_results:
            normalized_bm25 = r.score / max_bm25 if max_bm25 > 0 else 0
            
            all_chunks[r.chunk_id] = HybridResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=alpha * normalized_bm25,  # Will add vector later
                metadata=r.metadata,
                retrieval_method="hybrid_weighted",
                bm25_score=r.score,
                vector_score=0.0
            )
        
        # Process Vector results
        for r in vector_results:
            if r.chunk_id in all_chunks:
                # Update existing entry
                chunk = all_chunks[r.chunk_id]
                chunk.vector_score = r.score
                chunk.score += (1 - alpha) * r.score
            else:
                # New entry (only found by vector search)
                all_chunks[r.chunk_id] = HybridResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=(1 - alpha) * r.score,
                    metadata=r.metadata,
                    retrieval_method="hybrid_weighted",
                    bm25_score=0.0,
                    vector_score=r.score
                )
        
        return list(all_chunks.values())
    
    def _rrf_fusion(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        k: int = 60
    ) -> List[HybridResult]:
        """
        Reciprocal Rank Fusion (RRF).
        
        Formula: RRF_score = Σ 1/(k + rank)
        
        Benefits:
        - Doesn't require score normalization
        - Robust to score distribution differences
        - Works well when scores aren't comparable
        """
        all_chunks: Dict[str, HybridResult] = {}
        
        # Calculate RRF scores from BM25 rankings
        for rank, r in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (k + rank)
            
            if r.chunk_id in all_chunks:
                all_chunks[r.chunk_id].score += rrf_score
                all_chunks[r.chunk_id].bm25_score = r.score
            else:
                all_chunks[r.chunk_id] = HybridResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=rrf_score,
                    metadata=r.metadata,
                    retrieval_method="hybrid_rrf",
                    bm25_score=r.score,
                    vector_score=0.0
                )
        
        # Add RRF scores from Vector rankings
        for rank, r in enumerate(vector_results, 1):
            rrf_score = 1.0 / (k + rank)
            
            if r.chunk_id in all_chunks:
                all_chunks[r.chunk_id].score += rrf_score
                all_chunks[r.chunk_id].vector_score = r.score
            else:
                all_chunks[r.chunk_id] = HybridResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=rrf_score,
                    metadata=r.metadata,
                    retrieval_method="hybrid_rrf",
                    bm25_score=0.0,
                    vector_score=r.score
                )
        
        return list(all_chunks.values())
    
    def _max_fusion(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult]
    ) -> List[HybridResult]:
        """
        Max Score Fusion.
        
        Takes the maximum score from either method.
        Useful when you want to trust whichever method is more confident.
        """
        all_chunks: Dict[str, HybridResult] = {}
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results) or 1.0
        else:
            max_bm25 = 1.0
        
        # Process BM25 results
        for r in bm25_results:
            normalized_bm25 = r.score / max_bm25 if max_bm25 > 0 else 0
            
            all_chunks[r.chunk_id] = HybridResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=normalized_bm25,
                metadata=r.metadata,
                retrieval_method="hybrid_max",
                bm25_score=r.score,
                vector_score=0.0
            )
        
        # Process Vector results - take max
        for r in vector_results:
            if r.chunk_id in all_chunks:
                chunk = all_chunks[r.chunk_id]
                chunk.vector_score = r.score
                # Take maximum of the two scores
                normalized_bm25 = chunk.bm25_score / max_bm25 if max_bm25 > 0 else 0
                chunk.score = max(normalized_bm25, r.score)
            else:
                all_chunks[r.chunk_id] = HybridResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    retrieval_method="hybrid_max",
                    bm25_score=0.0,
                    vector_score=r.score
                )
        
        return list(all_chunks.values())
    
    def compare_methods(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare results from different retrieval methods.
        Useful for understanding which method works best for a query.
        """
        results = {
            "query": query,
            "bm25_only": [],
            "vector_only": [],
            "hybrid_weighted": [],
            "hybrid_rrf": []
        }
        
        # BM25 only
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        results["bm25_only"] = [
            {"id": r.chunk_id, "score": r.score, "preview": r.content[:100]}
            for r in bm25_results
        ]
        
        # Vector only
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k)
        results["vector_only"] = [
            {"id": r.chunk_id, "score": r.score, "preview": r.content[:100]}
            for r in vector_results
        ]
        
        # Hybrid weighted
        weighted_results = self.retrieve(query, top_k=top_k, method="weighted")
        results["hybrid_weighted"] = [
            {"id": r.chunk_id, "score": r.score, "bm25": r.bm25_score, 
             "vector": r.vector_score, "preview": r.content[:100]}
            for r in weighted_results
        ]
        
        # Hybrid RRF
        rrf_results = self.retrieve(query, top_k=top_k, method="rrf")
        results["hybrid_rrf"] = [
            {"id": r.chunk_id, "score": r.score, "preview": r.content[:100]}
            for r in rrf_results
        ]
        
        return results


if __name__ == "__main__":
    # Test Hybrid Retriever
    import sys
    import json
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import INDICES_DIR
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        bm25_index_path=INDICES_DIR / "bm25_index.pkl",
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        alpha=0.5
    )
    
    # Test queries
    queries = [
        "What is artificial intelligence?",
        "ML algorithms",
        "PRIMERGY server location",
        "How do computers learn from data?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: '{query}'")
        print("="*60)
        
        results = retriever.retrieve(query, top_k=3, method="weighted")
        
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Combined Score: {r.score:.3f}")
            print(f"    BM25: {r.bm25_score:.3f} | Vector: {r.vector_score:.3f}")
            print(f"    Content: {r.content[:80]}...")
    
    # Compare methods for one query
    print("\n" + "="*60)
    print("📊 Method Comparison")
    print("="*60)
    
    comparison = retriever.compare_methods("machine learning algorithms", top_k=3)
    print(json.dumps(comparison, indent=2, default=str)[:1000] + "...")
