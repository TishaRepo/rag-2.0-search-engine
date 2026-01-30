"""
NGSE Search Engine Service
Core business logic for the search engine pipeline.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import (
    INDICES_DIR, TOP_K_RETRIEVAL, TOP_K_RERANK, 
    HYBRID_ALPHA, RERANKER_MODEL, VECTOR_STORE_TYPE
)
from src.retrieval import HybridRetriever, Reranker
from src.reasoning import CoTReasoner, QueryDecomposer, Verifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchEngineService:
    """
    Core Search Engine Service.
    
    Orchestrates the full RAG 2.0 pipeline:
    1. Query Processing (decomposition if complex)
    2. Hybrid Retrieval (BM25 + Vector)
    3. Re-Ranking (Cross-Encoder)
    4. Reasoning (Chain-of-Thought)
    5. Verification (Hallucination Detection)
    """
    
    def __init__(
        self,
        bm25_index_path: Optional[Path] = None,
        chroma_persist_dir: Optional[str] = None,
        lazy_load: bool = True
    ):
        """
        Initialize the search engine service.
        
        Args:
            bm25_index_path: Path to BM25 index
            chroma_persist_dir: Path to ChromaDB directory
            lazy_load: If True, load components on first use
        """
        self.bm25_index_path = bm25_index_path or INDICES_DIR / "bm25_index.pkl"
        self.chroma_persist_dir = chroma_persist_dir or str(INDICES_DIR / "chroma")
        
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None
        self._reasoner: Optional[CoTReasoner] = None
        self._decomposer: Optional[QueryDecomposer] = None
        self._verifier: Optional[Verifier] = None
        
        if not lazy_load:
            self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing search engine components...")
        _ = self.retriever
        _ = self.reranker
        _ = self.reasoner
        logger.info("Search engine components initialized successfully")
    
    @property
    def retriever(self) -> HybridRetriever:
        """Lazy-load hybrid retriever."""
        if self._retriever is None:
            logger.info("Loading Hybrid Retriever...")
            self._retriever = HybridRetriever(
                bm25_index_path=self.bm25_index_path,
                chroma_persist_dir=self.chroma_persist_dir,
                alpha=HYBRID_ALPHA
            )
        return self._retriever
    
    @property
    def reranker(self) -> Reranker:
        """Lazy-load reranker."""
        if self._reranker is None:
            logger.info("Loading Reranker...")
            self._reranker = Reranker(model_name=RERANKER_MODEL)
        return self._reranker
    
    @property
    def reasoner(self) -> CoTReasoner:
        """Lazy-load CoT reasoner."""
        if self._reasoner is None:
            logger.info("Loading CoT Reasoner...")
            self._reasoner = CoTReasoner()
        return self._reasoner
    
    @property
    def decomposer(self) -> QueryDecomposer:
        """Lazy-load query decomposer."""
        if self._decomposer is None:
            logger.info("Loading Query Decomposer...")
            self._decomposer = QueryDecomposer()
        return self._decomposer
    
    @property
    def verifier(self) -> Verifier:
        """Lazy-load verifier."""
        if self._verifier is None:
            logger.info("Loading Verifier...")
            self._verifier = Verifier()
        return self._verifier
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K_RERANK,
        method: str = "hybrid",
        use_reranking: bool = True,
        use_reasoning: bool = True,
        use_verification: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the full search pipeline.
        
        Args:
            query: User search query
            top_k: Number of results to return
            method: Retrieval method (bm25, vector, hybrid)
            use_reranking: Whether to apply cross-encoder reranking
            use_reasoning: Whether to generate a reasoned answer
            use_verification: Whether to verify the answer
            
        Returns:
            Dictionary with search results and answer
        """
        start_time = time.time()
        
        result = {
            "query": query,
            "retrieved_documents": [],
            "answer": None,
            "reasoning": None,
            "verification": None,
        }
        
        try:
            # Step 1: Retrieval
            logger.info(f"Searching for: '{query[:50]}...'")
            retrieval_top_k = TOP_K_RETRIEVAL if use_reranking else top_k
            
            if method == "hybrid":
                retrieved = self.retriever.retrieve(query, top_k=retrieval_top_k)
            elif method == "bm25":
                retrieved = self.retriever.bm25_retriever.retrieve(query, top_k=retrieval_top_k)
            elif method == "vector":
                retrieved = self.retriever.vector_retriever.retrieve(query, top_k=retrieval_top_k)
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
            
            logger.info(f"Retrieved {len(retrieved)} documents")
            
            # Step 2: Reranking (optional)
            if use_reranking and retrieved:
                logger.info("Applying reranking...")
                retrieved = self.reranker.rerank(query, retrieved, top_k=top_k)
                logger.info(f"Reranked to top {len(retrieved)} documents")
            else:
                retrieved = retrieved[:top_k]
            
            # Convert to serializable format
            result["retrieved_documents"] = [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                    "retrieval_method": r.retrieval_method
                }
                for r in retrieved
            ]
            
            # Step 3: Reasoning (optional)
            if use_reasoning and retrieved:
                logger.info("Applying reasoning...")
                reasoning_result = self.reasoner.reason(query, retrieved)
                result["reasoning"] = {
                    "reasoning_steps": reasoning_result.get("reasoning_steps", []),
                    "final_answer": reasoning_result.get("final_answer", ""),
                    "source_count": reasoning_result.get("source_count", 0)
                }
                result["answer"] = reasoning_result.get("final_answer")
                
                # Step 4: Verification (optional)
                if use_verification and result["answer"]:
                    logger.info("Verifying answer...")
                    verification_result = self.verifier.verify(result["answer"], retrieved)
                    result["verification"] = {
                        "is_hallucination_suspected": verification_result.get("is_hallucination_suspected", False),
                        "confidence_score": verification_result.get("confidence_score", 0.0),
                        "claim_verifications": verification_result.get("claim_verifications", [])
                    }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            result["error"] = str(e)
        
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries."""
        return self.decomposer.decompose(query)
    
    def health_check(self) -> Dict[str, str]:
        """Check health of all components."""
        components = {}
        
        # Check retriever
        try:
            if VECTOR_STORE_TYPE == "pinecone":
                # Check Pinecone connection
                self.retriever.vector_retriever.index.describe_index_stats()
                components["retriever"] = "healthy (pinecone)"
            else:
                count = self.retriever.vector_retriever.collection.count()
                components["retriever"] = f"healthy (chroma, {count} documents)"
        except Exception as e:
            components["retriever"] = f"unhealthy: {e}"
        
        # Check reranker
        try:
            _ = self.reranker.model
            components["reranker"] = "healthy"
        except Exception as e:
            components["reranker"] = f"unhealthy: {e}"
        
        # Check LLM (reasoner)
        try:
            _ = self.reasoner.llm
            components["llm"] = "healthy"
        except Exception as e:
            components["llm"] = f"unhealthy: {e}"
        
        return components


# Singleton instance
_service_instance: Optional[SearchEngineService] = None


def get_search_service() -> SearchEngineService:
    """Get or create the singleton search service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SearchEngineService(lazy_load=True)
    return _service_instance
