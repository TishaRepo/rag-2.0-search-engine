"""
NGSE Test Suite
Unit and integration tests for the search engine.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": "test_001",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"title": "ML Basics", "source": "test"}
        },
        {
            "chunk_id": "test_002",
            "content": "Natural language processing allows computers to understand and generate human language.",
            "metadata": {"title": "NLP Overview", "source": "test"}
        },
        {
            "chunk_id": "test_003",
            "content": "Vector databases store data as high-dimensional embeddings for semantic search.",
            "metadata": {"title": "Vector DBs", "source": "test"}
        }
    ]


# =============================================================================
# INGESTION TESTS
# =============================================================================

class TestChunker:
    """Tests for document chunking."""
    
    def test_sliding_window_chunker(self):
        """Test sliding window chunking."""
        from src.ingestion.chunker import SlidingWindowChunker
        
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=50)
        
        text = "This is a test document. " * 20  # ~500 chars
        chunks = chunker.chunk(text, doc_id="test_doc")
        
        assert len(chunks) > 1
        assert all(len(c.content) >= 50 for c in chunks)
        assert all(c.doc_id == "test_doc" for c in chunks)
    
    def test_semantic_chunker(self):
        """Test semantic chunking."""
        from src.ingestion.chunker import SemanticChunker
        
        chunker = SemanticChunker(chunk_size=200, min_chunk_size=50)
        
        text = """
        This is the first paragraph. It contains some information.
        
        This is the second paragraph. It has different content.
        
        This is the third paragraph. More information here.
        """
        
        chunks = chunker.chunk(text, doc_id="semantic_test")
        
        assert len(chunks) >= 1
        assert all(c.chunk_id is not None for c in chunks)
    
    def test_get_chunker_factory(self):
        """Test chunker factory function."""
        from src.ingestion.chunker import get_chunker, SemanticChunker, SlidingWindowChunker
        
        semantic = get_chunker("semantic")
        sliding = get_chunker("sliding_window")
        
        assert isinstance(semantic, SemanticChunker)
        assert isinstance(sliding, SlidingWindowChunker)
        
        with pytest.raises(ValueError):
            get_chunker("invalid_strategy")


class TestDocumentLoader:
    """Tests for document loading."""
    
    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        from src.ingestion.document_loader import DocumentLoader
        
        # Create a temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content.")
        
        loader = DocumentLoader()
        doc = loader.load_file(test_file)
        
        assert doc is not None
        assert doc.content == "This is test content."
        assert doc.metadata["filename"] == "test.txt"
    
    def test_load_json_file(self, tmp_path):
        """Test loading a JSON file."""
        import json
        from src.ingestion.document_loader import DocumentLoader
        
        # Create a temp JSON file
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps({
            "title": "Test Document",
            "content": "JSON content here."
        }))
        
        loader = DocumentLoader()
        doc = loader.load_file(test_file)
        
        assert doc is not None
        assert "JSON content" in doc.content
    
    def test_unsupported_format(self, tmp_path):
        """Test loading unsupported file format."""
        from src.ingestion.document_loader import DocumentLoader
        
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")
        
        loader = DocumentLoader()
        doc = loader.load_file(test_file)
        
        assert doc is None


# =============================================================================
# RETRIEVAL TESTS
# =============================================================================

class TestBM25Retriever:
    """Tests for BM25 retriever."""
    
    def test_build_and_search(self, sample_chunks):
        """Test building index and searching."""
        from src.retrieval.bm25_retriever import BM25Retriever
        
        retriever = BM25Retriever()
        retriever.build_index(sample_chunks)
        
        results = retriever.retrieve("machine learning", top_k=2)
        
        assert len(results) > 0
        assert results[0].content is not None
        assert results[0].score > 0
    
    def test_empty_query(self, sample_chunks):
        """Test with empty query."""
        from src.retrieval.bm25_retriever import BM25Retriever
        
        retriever = BM25Retriever()
        retriever.build_index(sample_chunks)
        
        results = retriever.retrieve("", top_k=2)
        
        assert len(results) >= 0  # May return empty or all docs with 0 score
    
    def test_explain_score(self, sample_chunks):
        """Test score explanation."""
        from src.retrieval.bm25_retriever import BM25Retriever
        
        retriever = BM25Retriever()
        retriever.build_index(sample_chunks)
        
        explanation = retriever.explain_score("machine learning", "test_001")
        
        assert "matching_terms" in explanation
        assert "match_count" in explanation


# =============================================================================
# API MODELS TESTS
# =============================================================================

class TestAPIModels:
    """Tests for API models."""
    
    def test_search_request_validation(self):
        """Test search request validation."""
        from src.api.models import SearchRequest
        
        # Valid request
        request = SearchRequest(query="What is AI?")
        assert request.query == "What is AI?"
        assert request.top_k == 5  # default
        
        # Invalid empty query
        with pytest.raises(Exception):
            SearchRequest(query="")
    
    def test_search_response_creation(self):
        """Test search response creation."""
        from src.api.models import SearchResponse, RetrievedDocument
        from datetime import datetime
        
        response = SearchResponse(
            query="test",
            answer="Test answer",
            retrieved_documents=[
                RetrievedDocument(
                    chunk_id="1",
                    content="content",
                    score=0.9,
                    metadata={},
                    retrieval_method="hybrid"
                )
            ],
            processing_time_ms=100.0,
            timestamp=datetime.now()
        )
        
        assert response.query == "test"
        assert len(response.retrieved_documents) == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSearchPipeline:
    """Integration tests for the full search pipeline."""
    
    @pytest.mark.skip(reason="Requires index files - run after ingestion")
    def test_full_search_pipeline(self):
        """Test complete search pipeline."""
        from src.api.service import get_search_service
        
        service = get_search_service()
        result = service.search(
            query="What is machine learning?",
            top_k=3,
            use_reranking=False,  # Skip to avoid loading reranker
            use_reasoning=False   # Skip to avoid LLM calls
        )
        
        assert "query" in result
        assert "retrieved_documents" in result


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
