"""
NGSE Configuration Module
Centralized configuration for the Next-Gen Search Engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDICES_DIR = DATA_DIR / "indices"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDICES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

CHUNK_SIZE = 512  # Target tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep
MAX_CHUNK_SIZE = 1024  # Maximum chunk size

# Chunking strategy: "semantic" or "sliding_window"
CHUNKING_STRATEGY = "semantic"

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# Embedding model (from sentence-transformers)
# Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (better quality)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for MiniLM-L6-v2

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

# Number of documents to retrieve
TOP_K_RETRIEVAL = 20  # Initial retrieval
TOP_K_RERANK = 5  # After re-ranking

# Hybrid search weight (0 = pure vector, 1 = pure BM25)
HYBRID_ALPHA = 0.5

# =============================================================================
# RE-RANKING CONFIGURATION
# =============================================================================

# Cross-encoder model for re-ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# LLM Provider: "openai", "groq", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# API Keys (from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model names per provider
LLM_MODELS = {
    "openai": "gpt-4-turbo-preview",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-3-sonnet-20240229"
}

# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================

# ChromaDB settings
CHROMA_COLLECTION_NAME = "ngse_documents"
CHROMA_PERSIST_DIR = str(INDICES_DIR / "chroma")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ngse-search")
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma") # or "pinecone"

# =============================================================================
# VERIFICATION CONFIGURATION
# =============================================================================

# NLI model for fact-checking
NLI_MODEL = "facebook/bart-large-mnli"

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to accept a claim

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "ngse.log"

# Ensure log directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
