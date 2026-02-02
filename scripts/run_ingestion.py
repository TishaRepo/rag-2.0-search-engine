"""
Data Ingestion Pipeline
Run this script to ingest, chunk, and index documents.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, INDICES_DIR
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker
from src.ingestion.indexer import DocumentIndexer


def run_ingestion_pipeline():
    """Run the complete data ingestion pipeline."""
    
    print("=" * 60)
    print("🚀 NGSE Data Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load Documents
    print("\n📂 Step 1: Loading Documents...")
    loader = DocumentLoader()
    
    # Create sample corpus if no documents exist
    if not list(RAW_DATA_DIR.glob("*.json")):
        print("   Creating sample corpus...")
        docs = loader.create_sample_corpus(RAW_DATA_DIR)
    else:
        print("   Loading existing documents...")
        docs = loader.load_directory(RAW_DATA_DIR)
    
    print(f"   ✅ Loaded {len(docs)} documents")
    
    # Step 2: Chunk Documents
    print("\n✂️  Step 2: Chunking Documents...")
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
    
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(
            text=doc.content,
            doc_id=doc.doc_id,
            metadata=doc.metadata
        )
        all_chunks.extend(chunks)
    
    print(f"   ✅ Created {len(all_chunks)} chunks")
    
    # Step 3: Index Chunks
    print("\n🔍 Step 3: Indexing Chunks...")
    indexer = DocumentIndexer(
        embedding_model="all-MiniLM-L6-v2",
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        collection_name="ngse_documents"
    )
    
    stats = indexer.index_chunks(all_chunks)
    print(f"   ✅ Indexed {stats['indexed']} chunks")
    
    # Save BM25 index with document content
    indexer.save_bm25_index(INDICES_DIR / "bm25_index.pkl", chunks=all_chunks)
    print(f"   ✅ BM25 index saved")
    
    # Step 4: Test Search
    print("\n🔎 Step 4: Testing Search...")
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is RAG?",
        "Where is the PRIMERGY server?",
        "What is a ML?",
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = indexer.vector_search(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"   [{i}] Score: {r['score']:.3f}")
            print(f"       {r['content'][:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ Data Ingestion Complete!")
    print("=" * 60)
    
    return indexer


if __name__ == "__main__":
    run_ingestion_pipeline()
