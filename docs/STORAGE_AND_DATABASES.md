# 💾 Storage & Vector Databases - Complete Guide

## 📁 Where Are Your Documents Stored?

Here's your current storage structure:

```
data/
├── raw/                              ← 📄 YOUR ORIGINAL DOCUMENTS
│   ├── artificial_intelligence.json
│   ├── machine_learning.json
│   ├── natural_language_processing.json
│   ├── retrieval_augmented_generation.json
│   ├── vector_databases.json
│   └── asset-052018030468.txt        ← Your asset report!
│
├── processed/                        ← (Future: processed/cleaned data)
│
└── indices/                          ← 🔍 SEARCH INDICES
    │
    ├── bm25_index.pkl               ← BM25 keyword index (7KB)
    │
    └── chroma/                       ← VECTOR DATABASE
        ├── chroma.sqlite3           ← Metadata storage (397KB)
        └── 3e89467f-.../            ← Vector data folder
            ├── data_level0.bin      ← Actual vectors (167KB)
            ├── header.bin           ← Index header info
            ├── length.bin           ← Vector lengths
            └── link_lists.bin       ← HNSW graph links
```

---

## 📦 Understanding Each File Type

### 1️⃣ `.pkl` File (Pickle) - `bm25_index.pkl`

**What is it?**
- Python's way of saving objects to disk
- "Pickle" = serialize Python objects → save to file

**What's inside?**
```python
# Our BM25 index pickle contains:
{
    "corpus": [
        ["artificial", "intelligence", "is", ...],  # Tokenized doc 1
        ["machine", "learning", "ml", ...],         # Tokenized doc 2
        ...
    ],
    "chunk_ids": ["abc123", "def456", ...]          # IDs to map back
}
```

**How it's generated?**
```python
import pickle

data = {"corpus": [...], "chunk_ids": [...]}
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(data, f)  # Save to file
```

---

### 2️⃣ `.sqlite3` File - `chroma.sqlite3`

**What is it?**
- SQLite database (lightweight, file-based SQL database)
- ChromaDB uses this to store **metadata** (not vectors!)

**What's inside?**
```sql
-- Tables in chroma.sqlite3:
collections      -- Your collection "ngse_documents"
embeddings       -- Mapping: chunk_id → metadata
documents        -- Original text content
```

**Example data:**
| chunk_id | document | metadata |
|----------|----------|----------|
| abc123 | "AI is intelligence..." | {"title": "AI", "source": "..."} |
| def456 | "ML is a type of AI..." | {"title": "ML", "source": "..."} |

---

### 3️⃣ `.bin` Files (Binary) - The Actual Vectors!

**What are they?**

| File | Purpose |
|------|---------|
| `data_level0.bin` | **The actual vectors!** All 384-dim vectors stored as raw bytes |
| `header.bin` | Index configuration (dimension, capacity, etc.) |
| `length.bin` | Number of vectors at each level |
| `link_lists.bin` | HNSW graph connections (for fast search) |

**How vectors are stored:**
```
Each vector = 384 floats × 4 bytes = 1,536 bytes

data_level0.bin structure:
┌─────────────────────────────────────────────────────────┐
│ Vector 1: [0.12, -0.34, 0.56, ..., 0.23] (384 floats)   │
│ Vector 2: [0.45, 0.12, -0.78, ..., 0.67] (384 floats)   │
│ Vector 3: ...                                            │
└─────────────────────────────────────────────────────────┘
```

---

## 🗄️ What is ChromaDB?

### Overview

ChromaDB is an **open-source vector database** designed for AI applications.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ChromaDB                                 │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   SQLite    │    │   HNSW      │    │   Python API        │  │
│  │  (Metadata) │    │  (Vectors)  │    │   (Easy to use)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How ChromaDB Works Internally

#### Step 1: Adding Documents
```python
collection.add(
    ids=["chunk1"],
    embeddings=[[0.12, -0.34, ..., 0.23]],  # 384 numbers
    documents=["AI is intelligence..."],
    metadatas=[{"title": "AI"}]
)
```

What happens:
1. **SQLite**: Stores the document text and metadata
2. **HNSW Index**: Stores the vector in `.bin` files

#### Step 2: Searching
```python
results = collection.query(
    query_embeddings=[[0.15, -0.32, ..., 0.21]],
    n_results=5
)
```

What happens:
1. **HNSW Search**: Find 5 nearest vectors in `.bin` files
2. **SQLite Lookup**: Get documents/metadata for those vectors

### What is HNSW?

**HNSW = Hierarchical Navigable Small World**

It's the algorithm that makes vector search FAST!

```
Traditional Search:         HNSW Search:
Compare with ALL vectors    Navigate through layers
                           
O (●) (●) (●) (●) (●)      Level 2:  (●)───────(●)
  ↑   ↑   ↑   ↑   ↑                    ↓
  Check each one            Level 1:  (●)──(●)──(●)
  (slow for millions!)                  ↓
                           Level 0: (●)(●)(●)(●)(●)
                                         ↑
                           Found!        Jump to nearest
                           (Fast! O(log n))
```

---

## 🤔 Why ChromaDB? (vs Alternatives)

### Comparison Table

| Feature | ChromaDB | FAISS | Weaviate | Pinecone |
|---------|----------|-------|----------|----------|
| **Setup** | ⭐ Easiest | Medium | Complex | Cloud only |
| **Local/Cloud** | Local | Local | Both | Cloud only |
| **Cost** | Free | Free | Free/Paid | Paid |
| **Persistence** | Built-in | Manual | Built-in | Cloud |
| **Metadata** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Python API** | ⭐ Best | Good | Good | Good |
| **Best For** | Prototyping, Small-Medium | Maximum Speed | Production | Enterprise |

### Detailed Comparison

#### 🔵 FAISS (Facebook AI Similarity Search)
```python
# FAISS is FAST but requires more work
import faiss

index = faiss.IndexFlatL2(384)  # Create index
index.add(vectors)              # Add vectors
distances, indices = index.search(query, k=5)  # Search

# Problems:
# ❌ No built-in metadata storage (you manage separately)
# ❌ No persistence by default (you save/load manually)
# ❌ Just indices, no documents returned
```

**When to use**: Need maximum speed, millions of vectors, willing to manage complexity

#### 🟢 Weaviate
```python
# Weaviate is powerful but complex
import weaviate

client = weaviate.Client("http://localhost:8080")  # Needs server running!

client.schema.create_class({
    "class": "Document",
    "properties": [{"name": "content", "dataType": ["text"]}]
})

# Problems:
# ❌ Requires running a separate server (Docker)
# ❌ More complex schema definition
# ❌ Overkill for small projects
```

**When to use**: Production systems, need GraphQL API, complex schemas

#### 🟡 Pinecone
```python
# Pinecone is cloud-only
import pinecone

pinecone.init(api_key="your-key", environment="us-east-1")
index = pinecone.Index("my-index")

index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"metadata": "..."})])

# Problems:
# ❌ Cloud only (data leaves your machine)
# ❌ Costs money for production use
# ❌ Requires internet connection
```

**When to use**: Enterprise, don't want to manage infrastructure, have budget

#### 🟣 ChromaDB (Our Choice!)
```python
# ChromaDB is simple and complete
import chromadb

client = chromadb.PersistentClient(path="./data")  # One line!
collection = client.create_collection("docs")

collection.add(
    ids=["id1"],
    embeddings=[[0.1, 0.2, ...]],
    documents=["Original text"],      # ✅ Stores text!
    metadatas=[{"source": "file.txt"}]  # ✅ Stores metadata!
)

results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    include=["documents", "metadatas"]  # ✅ Returns everything!
)
```

**Why we chose ChromaDB:**
- ✅ **Zero setup** - Just `pip install chromadb`
- ✅ **Persistence built-in** - Saves automatically
- ✅ **Stores everything** - Vectors, documents, metadata
- ✅ **Simple API** - Easy to learn and use
- ✅ **Local** - Your data stays on your machine
- ✅ **Free** - Open source, no costs
- ✅ **Good for learning** - Perfect for understanding RAG

---

## 🔧 How Files Are Generated

### The Complete Flow

```python
# In indexer.py:

# 1. Initialize ChromaDB with persistence
client = chromadb.PersistentClient(path="data/indices/chroma")
#                                       ↓
#                        Creates: chroma.sqlite3
#                                 {uuid}/data_level0.bin, etc.

# 2. Create/get collection
collection = client.get_or_create_collection("ngse_documents")
#                                                    ↓
#                        Adds entry to: chroma.sqlite3 (collections table)

# 3. Add documents
collection.add(
    ids=["chunk1", "chunk2"],
    embeddings=[[...], [...]],  # 384-dim vectors
    documents=["text1", "text2"],
    metadatas=[{...}, {...}]
)
#     ↓
# chroma.sqlite3: Stores documents + metadata
# data_level0.bin: Stores vectors (raw bytes)
# link_lists.bin: Builds HNSW graph for fast search

# 4. Save BM25 index (separate from ChromaDB)
import pickle
with open("data/indices/bm25_index.pkl", "wb") as f:
    pickle.dump({"corpus": tokenized_docs, "chunk_ids": ids}, f)
```

---

## 📊 Size Comparison

For your 6 documents:

| File | Size | What's in it |
|------|------|--------------|
| `chroma.sqlite3` | 397 KB | Metadata, document text, mappings |
| `data_level0.bin` | 167 KB | 6 vectors × 384 dims × 4 bytes + overhead |
| `bm25_index.pkl` | 7 KB | Tokenized words + IDs |

**Scaling estimate:**
- 1,000 documents → ~30 MB
- 100,000 documents → ~3 GB
- 1,000,000 documents → ~30 GB (consider FAISS/Pinecone)

---

## 🎯 Summary

| Question | Answer |
|----------|--------|
| Where are documents? | `data/raw/` (originals), `data/indices/` (searchable) |
| What is `.pkl`? | Python pickle - serializes BM25 index |
| What is `.sqlite3`? | SQLite database - stores metadata & text |
| What are `.bin` files? | Raw binary vectors for HNSW search |
| Why ChromaDB? | Simple, local, free, stores everything, perfect for learning |
| When to switch? | FAISS for speed, Pinecone for cloud, Weaviate for production |

---

## 🔄 Want to Try FAISS Instead?

I can show you how to add FAISS as an alternative! It's already in your `requirements.txt`:
```
faiss-cpu
```

Just let me know if you want to see the comparison in code!
