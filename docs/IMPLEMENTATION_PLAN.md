# 🚀 NGSE Implementation Plan
## Next-Gen Search Engine (RAG 2.0 + Reasoning)

**Created**: 2026-01-07  
**Status**: In Progress  

---

## 📋 Executive Summary

This plan outlines the step-by-step implementation of a hallucination-resistant search engine using:
- **Hybrid Retrieval** (BM25 + Vector Search)
- **Chain-of-Thought Reasoning**
- **Hallucination Verification**
- **Source Attribution**

---

## 🗓️ Phase Timeline

| Phase | Name | Duration | Status |
|-------|------|----------|--------|
| 1 | Data Ingestion | Week 1 | ✅ Complete |
| 2 | Retrieval Layer | Week 2 | ✅ Complete |
| 3 | Re-Ranking | Week 2-3 | ✅ Complete |
| 4 | Reasoning Engine | Week 3-4 | ⏳ Pending |
| 5 | Verifier Model | Week 4 | ⏳ Pending |
| 6 | Answer Generation | Week 5 | ⏳ Pending |
| 7 | Evaluation | Week 5-6 | ⏳ Pending |

---

## 📦 PHASE 1: Data Ingestion (Complete)

### Objective
Create an indexed, searchable document store from a chosen corpus.

### Tasks

#### 1.1 Corpus Selection
- [x] Choose Wikipedia as primary corpus (accessible, diverse, well-structured)
- [x] Download Wikipedia sample dataset (100-1000 articles for testing)
- [x] Alternative: Support custom document upload (PDF, TXT, MD)

#### 1.2 Document Loader
- [x] Create `DocumentLoader` class
- [x] Support multiple formats: `.txt`, `.pdf`, `.md`, `.json`
- [x] Implement metadata extraction (title, source, date)

#### 1.3 Chunking Strategy
- [x] Implement **Semantic Chunking** (preferred)
- [x] Implement **Sliding Window Chunking** (fallback)
- [x] Add chunk metadata (source_doc, chunk_id, position)

#### 1.4 Vector Store Setup
- [x] Initialize ChromaDB for vector storage
- [x] Configure embedding model (`all-MiniLM-L6-v2`)
- [x] Create indexing pipeline

#### 1.5 BM25 Index
- [x] Build BM25 index for keyword search
- [x] Store alongside vector index

---

## 🔍 PHASE 2: Retrieval Layer (Complete)

### Objective
Build a hybrid retriever combining keyword (BM25) and semantic (vector) search.

### Tasks

#### 2.1 BM25 Retriever
- [x] Implement BM25 search using `rank_bm25`
- [x] Return top-K documents with scores

#### 2.2 Vector Retriever
- [x] Implement semantic search using ChromaDB
- [x] Use cosine similarity scoring
- [x] Return top-K documents with scores

#### 2.3 Hybrid Retrieval
- [x] Combine BM25 + Vector scores
- [x] Implement weighted fusion

---

## 🎯 PHASE 3: Re-Ranking (Complete)

### Objective
Re-rank retrieved documents using a cross-encoder for higher precision.

### Tasks

#### 3.1 Cross-Encoder Setup
- [x] Load cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [x] Score query-document pairs

#### 3.2 Re-Ranking Pipeline
- [x] Take top-N from hybrid retriever (N=10-20)
- [x] Re-score with cross-encoder
- [x] Return top-K (K=5)

### Deliverables
- `src/retrieval/reranker.py`

---

## 🧠 PHASE 4: Reasoning Engine

### Objective
Decompose complex queries and perform multi-hop reasoning.

### Tasks

#### 4.1 Query Decomposer
- [ ] Detect complex/multi-part queries
- [ ] Break into sub-queries using LLM
- [ ] Example: "Compare X and Y" → ["What is X?", "What is Y?", "Differences?"]

#### 4.2 Chain-of-Thought (CoT) Reasoning
- [ ] Implement step-by-step reasoning prompts
- [ ] Track reasoning chain for transparency

#### 4.3 Multi-Hop Retrieval
- [ ] Retrieve documents for each sub-query
- [ ] Aggregate and deduplicate results
- [ ] Build context from multiple sources

### Deliverables
- `src/reasoning/query_decomposer.py`
- `src/reasoning/cot_reasoner.py`

---

## ✅ PHASE 5: Verifier Model

### Objective
Detect and prevent hallucinations in generated answers.

### Tasks

#### 5.1 Fact Extraction
- [ ] Extract factual claims from LLM response
- [ ] Parse into verifiable statements

#### 5.2 Fact Validation
- [ ] Check each claim against retrieved documents
- [ ] Use NLI (Natural Language Inference) model
- [ ] Score: Supported / Contradicted / Neutral

#### 5.3 Confidence Scoring
- [ ] Aggregate claim scores
- [ ] Flag low-confidence answers
- [ ] Suggest verification for uncertain facts

### Deliverables
- `src/reasoning/verifier.py`
- `src/reasoning/fact_checker.py`

---

## 📝 PHASE 6: Answer Generation

### Objective
Generate concise, accurate answers with proper citations.

### Tasks

#### 6.1 Answer Synthesis
- [ ] Combine verified facts into coherent response
- [ ] Use LLM with context from retriever
- [ ] Enforce answer grounding in sources

#### 6.2 Citation Attachment
- [ ] Map answer sentences to source documents
- [ ] Add inline citations [1], [2], etc.
- [ ] Generate references section

#### 6.3 Response Formatting
- [ ] Structured output: Answer + Sources + Confidence
- [ ] Optional: Include reasoning trace

### Deliverables
- `src/reasoning/answer_generator.py`
- `src/utils/citation_formatter.py`

---

## 📊 PHASE 7: Evaluation

### Objective
Benchmark the system against vanilla RAG and measure quality metrics.

### Tasks

#### 7.1 Test Dataset
- [ ] Create evaluation dataset (100+ QA pairs)
- [ ] Include simple and complex queries
- [ ] Gold-standard answers for comparison

#### 7.2 Metrics Implementation
- [ ] **Faithfulness**: Do answers match sources?
- [ ] **Relevance**: Are answers on-topic?
- [ ] **Citation Accuracy**: Do citations support claims?
- [ ] **Hallucination Rate**: % of unsupported claims

#### 7.3 Benchmarking
- [ ] Compare vs vanilla RAG (no verification)
- [ ] Compare different retrieval strategies
- [ ] Measure latency and throughput

### Deliverables
- `src/evaluation/metrics.py`
- `src/evaluation/benchmark.py`
- `notebooks/evaluation_analysis.ipynb`

---

## 🛠️ Technical Specifications

### Models Used
| Component | Model | Purpose |
|-----------|-------|---------|
| Embeddings | `all-MiniLM-L6-v2` | Fast semantic embeddings |
| Cross-Encoder | `ms-marco-MiniLM-L-6-v2` | Re-ranking |
| NLI | `roberta-large-mnli` | Fact verification |
| LLM | GPT-4 / Claude / Groq | Reasoning & Generation |

### Storage
- **Vector DB**: ChromaDB (local, easy setup)
- **BM25 Index**: In-memory with pickle persistence
- **Documents**: JSON/Parquet in `data/`

### Configuration
All settings in `src/config.py`:
- Chunk size, overlap
- Retrieval K values
- Model paths
- API keys (via `.env`)

---

## 📁 Final Project Structure

```
rag 2.0 search engine/
├── data/
│   ├── raw/                 # Original documents
│   ├── processed/           # Chunked documents
│   └── indices/             # Vector + BM25 indices
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_testing.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── query_decomposer.py
│   │   ├── cot_reasoner.py
│   │   ├── verifier.py
│   │   └── answer_generator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── citation_formatter.py
│   │   └── helpers.py
│   ├── config.py
│   └── main.py
├── tests/
├── .env
├── requirements.txt
└── README.md
```

---

## 🎯 Next Steps (Immediate)

1. ✅ Created implementation plan
2. 🔄 Build Phase 1 data ingestion components
3. 📥 Download sample Wikipedia dataset
4. 🧪 Test chunking and indexing pipeline

---

*This plan will be updated as we progress through each phase.*
