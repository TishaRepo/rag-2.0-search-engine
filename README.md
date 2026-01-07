# NGSE: Next-Gen Search Engine (RAG 2.0 + Reasoning)

## 🎯 Goal
Build a hallucination-resistant search engine using Retrieval, Reasoning, Verification, and Source attribution.

## 🧱 System Architecture
`User Query` -> `Query Decomposer` -> `Retriever` -> `Re-Ranker` -> `LLM Reasoner` -> `Verifier` -> `Final Answer`


### PHASE 1: Data Ingestion
- [ ] Choose corpus (Wikipedia, ArXiv, etc.)
- [ ] Implement Chunking Strategy (Semantic, Sliding Window)
- **Deliverable**: Indexed document store

### PHASE 2: Retrieval Layer
- [ ] BM25 (Keyword) implementation
- [ ] Vector Search (Semantic) implementation
- [ ] Hybrid Retrieval (Weighted scoring)
- **Deliverable**: Hybrid retriever module

### PHASE 3: Re-Ranking
- [ ] Cross-encoder scoring
- [ ] Filter top-K documents
- **Deliverable**: Re-ranker model

### PHASE 4: Reasoning Engine
- [ ] Query Decomposition
- [ ] Multi-hop reasoning (CoT, Planning)
- **Deliverable**: Reasoning pipeline

### PHASE 5: Verifier Model
- [ ] Hallucination detection
- [ ] Fact validation
- **Deliverable**: Confidence scoring

### PHASE 6: Answer Generation
- [ ] Concise answer generation
- [ ] Citation attachment
- **Deliverable**: Final Answer with Sources

### PHASE 7: Evaluation
- [ ] Faithfulness & Relevance metrics
- [ ] Benchmark vs Vanilla RAG
