---
title: RAG Search Engine
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

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

The API will be available at `http://localhost:8000/ui`
- Swagger Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`


```





## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Groq](https://groq.com/) for fast LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
