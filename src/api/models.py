"""
NGSE API Models (Pydantic Schemas)
Request and response models for the FastAPI application.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST MODELS
# =============================================================================

class SearchRequest(BaseModel):
    """Search query request."""
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    method: str = Field(default="hybrid", description="Retrieval method: bm25, vector, or hybrid")
    use_reranking: bool = Field(default=True, description="Whether to use cross-encoder reranking")
    use_reasoning: bool = Field(default=True, description="Whether to use CoT reasoning for answer generation")
    use_verification: bool = Field(default=False, description="Whether to verify the answer for hallucinations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning and how does it work?",
                "top_k": 5,
                "method": "hybrid",
                "use_reranking": True,
                "use_reasoning": True,
                "use_verification": False
            }
        }


class IngestRequest(BaseModel):
    """Document ingestion request."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to ingest")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, le=200, description="Overlap between chunks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "title": "Machine Learning Basics",
                        "content": "Machine learning is a subset of artificial intelligence..."
                    }
                ],
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }


class QueryDecomposeRequest(BaseModel):
    """Query decomposition request."""
    query: str = Field(..., min_length=1, max_length=1000)


class DeleteRequest(BaseModel):
    """Document deletion request."""
    ids: Optional[List[str]] = Field(None, description="List of document IDs to delete")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter for deletion")
    delete_all: bool = Field(default=False, description="Whether to clear the entire index")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class RetrievedDocument(BaseModel):
    """A single retrieved document chunk."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str


class ReasoningResult(BaseModel):
    """Reasoning engine output."""
    reasoning_steps: List[str] = []
    final_answer: str
    source_count: int


class VerificationResult(BaseModel):
    """Verification/hallucination check output."""
    is_hallucination_suspected: bool
    confidence_score: float
    claim_verifications: List[Dict[str, Any]] = []


class SearchResponse(BaseModel):
    """Complete search response."""
    query: str
    answer: Optional[str] = None
    reasoning: Optional[ReasoningResult] = None
    verification: Optional[VerificationResult] = None
    retrieved_documents: List[RetrievedDocument] = []
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI that enables systems to learn from data...",
                "reasoning": {
                    "reasoning_steps": ["Step 1: Analyzed context...", "Step 2: Identified key facts..."],
                    "final_answer": "Machine learning is...",
                    "source_count": 3
                },
                "retrieved_documents": [],
                "processing_time_ms": 1234.56,
                "timestamp": "2024-01-28T12:00:00Z"
            }
        }


class IngestResponse(BaseModel):
    """Document ingestion response."""
    status: str
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]


class QueryDecomposeResponse(BaseModel):
    """Query decomposition response."""
    original_query: str
    sub_queries: List[str]


class DeleteResponse(BaseModel):
    """Document deletion response."""
    status: str
    deleted_count: int
    message: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
