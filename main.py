"""
NGSE: Next-Gen Search Engine
Main FastAPI Application Entry Point

Usage:
    Development:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
    Production:   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import __version__
from src.api import router
from src.api.service import get_search_service
from src.utils.logger import setup_logging, get_logger
from src.config import LOG_LEVEL, LOG_FILE

# Setup logging
setup_logging(
    level=LOG_LEVEL,
    log_file=LOG_FILE,
    json_format=os.getenv("ENVIRONMENT", "development") == "production"
)

logger = get_logger(__name__)


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Manages startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"🚀 Starting NGSE v{__version__}")
    logger.info("=" * 60)
    
    # Pre-load components if PRELOAD_MODELS is set
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        logger.info("Pre-loading ML models...")
        try:
            service = get_search_service()
            _ = service.retriever  # Load retriever
            _ = service.reranker   # Load reranker
            logger.info("✅ Models pre-loaded successfully")
        except Exception as e:
            logger.error(f"⚠️ Failed to pre-load models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NGSE...")
    logger.info("Goodbye! 👋")


# =============================================================================
# CREATE APPLICATION
# =============================================================================

app = FastAPI(
    title="NGSE - Next-Gen Search Engine",
    description="""
    ## A Hallucination-Resistant RAG 2.0 Search Engine
    
    NGSE combines multiple techniques to provide accurate, verifiable answers:
    
    - **Hybrid Retrieval**: BM25 (keyword) + Vector (semantic) search
    - **Cross-Encoder Reranking**: Improved precision with neural rerankers
    - **Chain-of-Thought Reasoning**: Step-by-step answer generation
    - **Hallucination Detection**: Fact verification against sources
    
    ### Quick Start
    
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/search",
        json={
            "query": "What is machine learning?",
            "top_k": 5,
            "use_reasoning": True
        }
    )
    print(response.json()["answer"])
    ```
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS - Configure for your production domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )


# =============================================================================
# ROUTES
# =============================================================================

# Include API routes
app.include_router(router, prefix="/api/v1")

# Also mount at root for convenience
app.include_router(router)

# Mount static files
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve web UI at root
@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def web_ui():
    """Serve the web search interface."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UI not found. Check static/index.html</h1>", status_code=404)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level="info"
    )
