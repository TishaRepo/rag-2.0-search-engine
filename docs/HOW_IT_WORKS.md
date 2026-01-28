# 🧠 How the RAG Search Engine Works - Complete Guide

## 📖 Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Step-by-Step Flow](#step-by-step-flow)
3. [Understanding Each Component](#understanding-each-component)
4. [Code Walkthrough](#code-walkthrough)
5. [Real Example](#real-example)

---

## 🎯 The Big Picture

### What Problem Does This Solve?

Imagine you have **1000 documents** and you want to ask:
> "Where is the PRIMERGY server located?"

**Without this system**: You'd have to open and read every document manually.

**With this system**: You type your question, and it instantly finds the relevant documents and shows you the answer.

### How Does It Work? (Simple Version)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   YOUR       │     │   BREAK      │     │   STORE      │     │   SEARCH     │
│   DOCUMENTS  │ ──► │   INTO       │ ──► │   IN         │ ──► │   AND        │
│              │     │   PIECES     │     │   DATABASE   │     │   FIND       │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
    (Input)            (Chunking)           (Indexing)          (Retrieval)
```

**Think of it like a library:**
1. **Documents** = Books in your library
2. **Chunking** = Creating an index card for each important section
3. **Indexing** = Organizing cards in a searchable catalog
4. **Retrieval** = Finding cards that match your question

---

## 🔄 Step-by-Step Flow

### PHASE 1: Data Ingestion (What Happens When You Run `run_ingestion.py`)

```
Your Files                          What the System Does
─────────────────────────────────────────────────────────────────────

data/raw/
├── artificial_intelligence.json    ──┐
├── machine_learning.json             │
├── natural_language_processing.json  ├──► DocumentLoader reads all files
├── retrieval_augmented_generation.json│
├── vector_databases.json             │
└── asset-052018030468.txt          ──┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  6 Documents    │
                              │  loaded into    │
                              │  memory         │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Chunker splits │
                              │  each document  │──► Creates 7 chunks
                              │  into pieces    │    (smaller pieces)
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Indexer stores │
                              │  chunks in two  │
                              │  databases      │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
           ┌─────────────────┐                    ┌─────────────────┐
           │  ChromaDB       │                    │  BM25 Index     │
           │  (Vector Store) │                    │  (Keyword Store)│
           │                 │                    │                 │
           │  Stores the     │                    │  Stores word    │
           │  "meaning" of   │                    │  frequencies    │
           │  each chunk     │                    │  for keyword    │
           └─────────────────┘                    │  matching       │
                                                  └─────────────────┘
```

---

## 🧩 Understanding Each Component

### 1️⃣ Document Loader (`document_loader.py`)

**Purpose**: Read files from your `data/raw/` folder

**What it does**:
```python
# Simplified logic:
for each file in data/raw/:
    if file is .txt:
        content = read the text
    if file is .json:
        content = extract "text" field
    
    create Document(content, metadata)
```

**Example**:
```
Input:  asset-052018030468.txt
Output: Document(
            content="ASSET DETAILS REPORT... PRIMERGY TX1310...",
            metadata={"filename": "asset-052018030468.txt", "source": "..."}
        )
```

---

### 2️⃣ Chunker (`chunker.py`)

**Purpose**: Break big documents into smaller, searchable pieces

**Why chunk?**
- Large documents are hard to search effectively
- Smaller chunks = more precise results
- LLMs have token limits

**What it does**:
```python
# Simplified logic:
def chunk(document):
    pieces = []
    current_piece = ""
    
    for paragraph in document:
        if len(current_piece) + len(paragraph) > 500:  # chunk size limit
            pieces.append(current_piece)
            current_piece = paragraph
        else:
            current_piece += paragraph
    
    return pieces
```

**Example**:
```
Input:  "Artificial intelligence is... [1000 words] ...neural networks."

Output: [
    Chunk 1: "Artificial intelligence is... [first 500 chars]",
    Chunk 2: "...machine learning... [next 500 chars]",
    Chunk 3: "...neural networks. [remaining text]"
]
```

---

### 3️⃣ Indexer (`indexer.py`)

**Purpose**: Store chunks so they can be searched quickly

**Two types of indexing**:

#### A) Vector Index (ChromaDB) - "Meaning-based search"

**How it works**:
1. Each chunk is converted to a **vector** (list of 384 numbers)
2. Similar meanings = similar vectors
3. When you search, your query is also converted to a vector
4. Find chunks whose vectors are "close" to your query vector

```python
# Simplified:
"Machine learning is AI"     → [0.23, -0.45, 0.12, ..., 0.67]  # 384 numbers
"AI learns from data"        → [0.21, -0.43, 0.14, ..., 0.65]  # Similar!
"Pizza is delicious"         → [0.89, 0.12, -0.56, ..., 0.01]  # Different!
```

**Why vectors?**
- "ML" and "Machine Learning" have similar vectors (same meaning!)
- Keyword search would miss this connection

#### B) BM25 Index - "Keyword-based search"

**How it works**:
1. Count how often each word appears in each chunk
2. When you search for "PRIMERGY", find chunks with that word
3. Rank by: how often word appears × how rare the word is globally

```python
# Simplified:
Chunk 1: {"primergy": 2, "server": 3, "located": 1}
Chunk 2: {"ai": 5, "learning": 4, "machine": 3}

Search "primergy" → Chunk 1 wins (has the word)
```

---

## 💻 Code Walkthrough

### What Happens When You Run `run_ingestion.py`:

```python
# Step 1: Load all documents from data/raw/
loader = DocumentLoader()
docs = loader.load_directory("data/raw/")
# Result: 6 Document objects

# Step 2: Split each document into chunks
chunker = SemanticChunker(chunk_size=500)
all_chunks = []
for doc in docs:
    chunks = chunker.chunk(doc.content)  # Split into ~500 char pieces
    all_chunks.extend(chunks)
# Result: 7 Chunk objects

# Step 3: Index all chunks
indexer = DocumentIndexer()
indexer.index_chunks(all_chunks)
# This does:
#   a) Convert each chunk to a 384-dim vector using AI model
#   b) Store vectors in ChromaDB
#   c) Build BM25 keyword index
#   d) Save everything to disk
```

### What Happens When You Search:

```python
# Step 1: Load the existing indices
indexer = DocumentIndexer(chroma_persist_dir="data/indices/chroma")
indexer.load_bm25_index("data/indices/bm25_index.pkl")

# Step 2: Convert your question to a vector
query = "Where is the PRIMERGY server?"
query_vector = embedding_model.encode(query)  # → [0.34, -0.12, ...]

# Step 3: Find similar chunks
# Vector search: find chunks with similar meaning
vector_results = chromadb.find_similar(query_vector, top_k=5)

# BM25 search: find chunks with matching words  
bm25_results = bm25_index.search("primergy server", top_k=5)

# Step 4: Return the best matches
```

---

## 📋 Real Example

### Your Asset Document:

```
ASSET DETAILS REPORT
====================
Asset Number: 052018030468
Product Name: PRIMERGY TX1310
Building: High Throughput Building
Floor: 2F
Room: 208
```

### After Ingestion:

The system creates:
```
Chunk ID: abc123
Content: "ASSET DETAILS REPORT Asset Number: 052018030468 
          Product Name: PRIMERGY TX1310 Building: High 
          Throughput Building Floor: 2F Room: 208..."
Vector: [0.12, -0.34, 0.56, ..., 0.23]  # 384 numbers representing meaning
```

### When You Search "Where is the PRIMERGY server?":

```
1. Query converted to vector: [0.15, -0.32, 0.54, ..., 0.21]

2. Vector Search:
   - Compare query vector to all chunk vectors
   - Chunk abc123 has similarity score: 0.87 (high match!)
   - Returns: "PRIMERGY TX1310... Building: High Throughput..."

3. BM25 Search:
   - Look for "primergy" and "server" in chunks
   - Chunk abc123 contains "PRIMERGY" → Match!
   - Returns: "PRIMERGY TX1310... Building: High Throughput..."

4. Final Answer: 
   "The PRIMERGY server is in High Throughput Building, 2F, Room 208"
```

---

## 📁 File Summary

| File | What It Does |
|------|-------------|
| `document_loader.py` | Reads .txt, .json, .md files into memory |
| `chunker.py` | Splits documents into ~500 character pieces |
| `indexer.py` | Stores chunks in Vector DB + BM25 index |
| `run_ingestion.py` | Runs the complete pipeline |
| `config.py` | Settings (chunk size, model names, paths) |

---

## 🔑 Key Concepts

| Term | Meaning |
|------|---------|
| **Document** | One file (txt, json, md) |
| **Chunk** | A piece of a document (~500 chars) |
| **Embedding/Vector** | 384 numbers representing "meaning" |
| **ChromaDB** | Database that stores vectors |
| **BM25** | Algorithm for keyword matching |
| **Similarity Score** | How close two vectors are (0-1) |

---

## 🎯 Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THE COMPLETE FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📄 Documents  ──►  ✂️ Chunks  ──►  🔢 Vectors  ──►  💾 Database    │
│                                                                     │
│  When you search:                                                   │
│                                                                     │
│  ❓ Question  ──►  🔢 Query Vector  ──►  🔍 Find Similar  ──►  📋   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**In simple terms:**
1. Your documents are broken into small pieces
2. Each piece is converted to numbers representing its meaning
3. When you ask a question, we find pieces with similar numbers
4. Those pieces contain your answer!

---

*This is Phase 1 (Data Ingestion). Later phases will add:*
- *Hybrid search (combine vector + keyword)*
- *Re-ranking (improve result order)*
- *Reasoning (answer complex questions)*
- *Verification (prevent wrong answers)*
