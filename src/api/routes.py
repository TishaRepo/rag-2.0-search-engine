"""
NGSE API Routes
FastAPI endpoints for the search engine.
"""

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src import __version__
from src.api.models import (
    SearchRequest, SearchResponse, RetrievedDocument,
    ReasoningResult, VerificationResult,
    IngestRequest, IngestResponse,
    HealthResponse, ErrorResponse,
    QueryDecomposeRequest, QueryDecomposeResponse
)
from src.api.service import get_search_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns the status of all system components.
    """
    try:
        service = get_search_service()
        components = service.health_check()
        
        # Determine overall status
        overall_status = "healthy" if all("healthy" in v for v in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version=__version__,
            timestamp=datetime.now(),
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            timestamp=datetime.now(),
            components={"error": str(e)}
        )


@router.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NGSE - Next-Gen Search Engine",
        "version": __version__,
        "description": "A hallucination-resistant RAG 2.0 search engine",
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================

@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Execute a search query through the RAG 2.0 pipeline.
    
    The pipeline includes:
    1. Hybrid Retrieval (BM25 + Vector)
    2. Cross-Encoder Reranking
    3. Chain-of-Thought Reasoning
    4. Hallucination Verification (optional)
    
    Returns retrieved documents and an AI-generated answer with sources.
    """
    start_time = time.time()
    
    try:
        service = get_search_service()
        
        result = service.search(
            query=request.query,
            top_k=request.top_k,
            method=request.method,
            use_reranking=request.use_reranking,
            use_reasoning=request.use_reasoning,
            use_verification=request.use_verification
        )
        
        # Build response
        retrieved_docs = [
            RetrievedDocument(**doc) for doc in result.get("retrieved_documents", [])
        ]
        
        reasoning = None
        if result.get("reasoning"):
            reasoning = ReasoningResult(**result["reasoning"])
        
        verification = None
        if result.get("verification"):
            verification = VerificationResult(**result["verification"])
        
        return SearchResponse(
            query=request.query,
            answer=result.get("answer"),
            reasoning=reasoning,
            verification=verification,
            retrieved_documents=retrieved_docs,
            processing_time_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    method: str = Query(default="hybrid", description="Retrieval method"),
    rerank: bool = Query(default=True, description="Use reranking"),
    reason: bool = Query(default=True, description="Use reasoning")
):
    """GET endpoint for simple search queries."""
    request = SearchRequest(
        query=q,
        top_k=top_k,
        method=method,
        use_reranking=rerank,
        use_reasoning=reason
    )
    return await search(request)


# =============================================================================
# QUERY PROCESSING ENDPOINTS
# =============================================================================

@router.post("/decompose", response_model=QueryDecomposeResponse, tags=["Query Processing"])
async def decompose_query(request: QueryDecomposeRequest):
    """
    Decompose a complex query into simpler sub-queries.
    
    Useful for multi-hop reasoning and understanding complex questions.
    """
    try:
        service = get_search_service()
        sub_queries = service.decompose_query(request.query)
        
        return QueryDecomposeResponse(
            original_query=request.query,
            sub_queries=sub_queries
        )
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DATA MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/ingest", response_model=IngestResponse, tags=["Data Management"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest new documents into the search index.
    
    Documents are chunked and indexed for retrieval.
    """
    start_time = time.time()
    
    try:
        from src.ingestion import DocumentLoader, SemanticChunker, DocumentIndexer
        from src.config import INDICES_DIR
        
        # Create chunks
        chunker = SemanticChunker(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        all_chunks = []
        for doc in request.documents:
            from src.ingestion.document_loader import Document
            
            # Extract content
            content = doc.get("content", doc.get("text", ""))
            
            # Prepare metadata (flatten any nested structures)
            # We skip 'content' and 'text' to avoid redundancy and size limits
            clean_metadata = {}
            for k, v in doc.items():
                if k in ["content", "text"]:
                    continue
                if isinstance(v, dict):
                    # Flatten nested dict: {"metadata": {"source": "wiki"}} -> {"source": "wiki"}
                    for sub_k, sub_v in v.items():
                        if not isinstance(sub_v, (dict, list)):
                            clean_metadata[sub_k] = sub_v
                elif not isinstance(v, list):
                    clean_metadata[k] = v
            
            document = Document(
                content=content,
                metadata=clean_metadata
            )
            
            chunks = chunker.chunk(
                document.content,
                doc_id=doc.get("id") or doc.get("doc_id"),
                metadata=document.metadata
            )
            all_chunks.extend(chunks)
        
        # Index chunks
        indexer = DocumentIndexer(
            chroma_persist_dir=str(INDICES_DIR / "chroma"),
            collection_name="ngse_documents"
        )
        stats = indexer.index_chunks(all_chunks)
        indexer.save_bm25_index(INDICES_DIR / "bm25_index.pkl", chunks=all_chunks)
        
        return IngestResponse(
            status="success",
            documents_processed=len(request.documents),
            chunks_created=len(all_chunks),
            chunks_indexed=stats.get("indexed", 0),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", tags=["Data Management"])
async def get_stats():
    """Get statistics about the indexed documents."""
    try:
        service = get_search_service()
        
        vector_count = service.retriever.vector_retriever.collection.count()
        bm25_count = len(service.retriever.bm25_retriever.documents)
        
        return {
            "vector_documents": vector_count,
            "bm25_documents": bm25_count,
            "embedding_model": service.retriever.vector_retriever.embedding_model_name,
            "reranker_model": service.reranker.model.model.name_or_path if hasattr(service.reranker.model, 'model') else "unknown"
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
