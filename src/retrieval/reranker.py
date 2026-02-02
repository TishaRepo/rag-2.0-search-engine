"""
Reranker Module
Uses a Cross-Encoder to refine the order of retrieved documents.
"""

import logging
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder

from .bm25_retriever import RetrievalResult

logger = logging.getLogger(__name__)

class Reranker:
    """
    Cross-Encoder Reranker.
    
    Why use a Reranker?
    1. Retrieval (Phase 2) is fast but "coarse". It looks at similarity in a vector space.
    2. Reranking (Phase 3) is slower but "fine-grained". It looks at the query and document 
       together, calculating a much more accurate relevance score.
       
    The Pipeline:
    Retrieve 20 docs (Hybrid) -> Rerank those 20 -> Keep top 5.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (optimized for search)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Reranker.
        
        Args:
            model_name: The name of the Cross-Encoder model to use.
        """
        logger.info(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully.")

    def rerank(self, query: str, results: List[RetrievalResult], top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Re-score and re-sort a list of retrieval results.
        
        Args:
            query: The user query.
            results: List of RetrievalResult objects from the retriever.
            top_k: Number of results to return after reranking.
            
        Returns:
            List of RetrievalResult objects sorted by the reranker's score.
        """
        if not results:
            return []

        # Prepare pairs for the Cross-Encoder: (query, document_text)
        pairs = [[query, r.content] for r in results]
        
        # Get scores
        logger.debug(f"Reranking {len(results)} pairs...")
        scores = self.model.predict(pairs)
        
        # Update scores and reranking method in results
        reranked_results = []
        for r, score in zip(results, scores):
            # Create a copy or update the object
            r.score = float(score)
            r.retrieval_method += "+rerank"
            reranked_results.append(r)
            
        # Sort by the new score (higher is better)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        if top_k:
            return reranked_results[:top_k]
        
        return reranked_results
