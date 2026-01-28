# 🔍 NGSE: Next-Gen Search Engine (RAG 2.0 + Reasoning)

A **hallucination-resistant search engine** using Retrieval-Augmented Generation with reasoning and verification capabilities.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- **Hybrid Retrieval**: Combines BM25 (keyword) and Vector (semantic) search for best results
- **Cross-Encoder Reranking**: Neural reranker for improved precision
- **Chain-of-Thought Reasoning**: Step-by-step answer generation with source attribution
- **Hallucination Detection**: LLM-based verification against retrieved sources
- **Production Ready**: FastAPI, Docker, health checks, structured logging

## 🏗️ System Architecture

```
User Query → Query Decomposer → Hybrid Retriever → Re-Ranker → LLM Reasoner → Verifier → Final Answer
                    ↓                   ↓                           ↓
              Sub-queries          BM25 + Vector              Chain-of-Thought
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/) or [OpenAI API Key](https://platform.openai.com/)

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/ngse.git
cd ngse

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys

# Run data ingestion (first time)
python scripts/run_ingestion.py

# Start the server
python main.py
```

The API will be available at `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# With nginx reverse proxy (production)
docker-compose --profile production up -d

# View logs
docker-compose logs -f ngse-api
```

## 📡 API Endpoints

### Search
```bash
# POST request
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "use_reasoning": true
  }'

# GET request (simple)
curl "http://localhost:8000/search?q=What%20is%20AI&top_k=3"
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Document Ingestion
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"title": "My Document", "content": "Document content here..."}
    ]
  }'
```

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider: groq, openai, anthropic |
| `GROQ_API_KEY` | - | Your Groq API key |
| `OPENAI_API_KEY` | - | Your OpenAI API key |
| `ENVIRONMENT` | `development` | Environment mode |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HYBRID_ALPHA` | `0.5` | BM25 vs Vector weight (0-1) |
| `PRELOAD_MODELS` | `false` | Pre-load ML models on startup |

## 📁 Project Structure

```
ngse/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker build configuration
├── docker-compose.yml     # Docker Compose setup
├── .env.example           # Environment template
├── src/
│   ├── config.py          # Centralized configuration
│   ├── api/               # FastAPI routes and models
│   │   ├── routes.py      # API endpoints
│   │   ├── models.py      # Pydantic schemas
│   │   └── service.py     # Business logic
│   ├── ingestion/         # Document processing
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── retrieval/         # Search components
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── reasoning/         # Answer generation
│   │   ├── query_decomposer.py
│   │   ├── cot_reasoner.py
│   │   └── verifier.py
│   └── utils/             # Utilities
│       ├── llm_client.py
│       └── logger.py
├── scripts/               # Utility scripts
│   └── run_ingestion.py
├── data/                  # Data storage
│   ├── raw/              # Source documents
│   ├── processed/        # Processed data
│   └── indices/          # Search indices
└── docs/                  # Documentation
```

## 🚢 Production Deployment

### Option 1: Docker (Recommended)

```bash
# Build production image
docker build -t ngse:latest .

# Run with environment variables
docker run -d \
  --name ngse \
  -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e ENVIRONMENT=production \
  -v ngse-data:/app/data \
  ngse:latest
```

### Option 2: Cloud Platforms

#### AWS ECS / Google Cloud Run / Azure Container Apps

1. Push image to your container registry
2. Configure environment variables
3. Set up health check endpoint: `/health`
4. Configure memory: minimum 2GB, recommended 4GB

#### Railway / Render / Fly.io

```bash
# Example: Railway
railway login
railway init
railway up
```

### Option 3: Traditional Server

```bash
# Install with Gunicorn
pip install gunicorn uvicorn

# Run with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest scripts/test_retrieval.py
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint
flake8 src/
mypy src/
```

## 📊 Performance Tips

1. **Pre-load models**: Set `PRELOAD_MODELS=true` to avoid cold start latency
2. **Use GPU**: For faster embeddings, install `torch` with CUDA support
3. **Increase workers**: Use `gunicorn -w 4` for multi-core utilization
4. **Rate limiting**: Use nginx for production rate limiting
5. **Caching**: Consider Redis for frequently accessed queries

## 🛠️ Troubleshooting

### Common Issues

**"GROQ_API_KEY not found"**
- Ensure `.env` file exists with valid API key
- Check that python-dotenv is loading the file

**"BM25 index not found"**
- Run `python scripts/run_ingestion.py` to create indices

**"Out of memory"**
- Reduce batch size in indexer
- Use a smaller embedding model

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Groq](https://groq.com/) for fast LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
