"""
Test Hybrid Retrieval
Demonstrates and compares BM25, Vector, and Hybrid search.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INDICES_DIR
from src.retrieval import BM25Retriever, VectorRetriever, HybridRetriever


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_results(results, method_name: str):
    """Print search results in a nice format."""
    print(f"\n📊 {method_name} Results:")
    print("-" * 50)
    
    if not results:
        print("  No results found.")
        return
    
    for i, r in enumerate(results, 1):
        score_str = f"Score: {r.score:.3f}"
        
        # Add extra info for hybrid results
        if hasattr(r, 'bm25_score') and hasattr(r, 'vector_score'):
            score_str += f" (BM25: {r.bm25_score:.2f}, Vector: {r.vector_score:.2f})"
        
        print(f"\n  [{i}] {score_str}")
        print(f"      {r.content[:100]}...")


def main():
    print_header("🔍 NGSE Hybrid Retrieval Demo")
    
    # Initialize retrievers
    print("\n⏳ Loading retrievers...")
    
    bm25_retriever = BM25Retriever(index_path=INDICES_DIR / "bm25_index.pkl")
    print("  ✅ BM25 Retriever loaded")
    
    vector_retriever = VectorRetriever(
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        collection_name="ngse_documents"
    )
    print("  ✅ Vector Retriever loaded")
    
    hybrid_retriever = HybridRetriever(
        bm25_index_path=INDICES_DIR / "bm25_index.pkl",
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        alpha=0.5  # Equal weight to BM25 and Vector
    )
    print("  ✅ Hybrid Retriever loaded")
    
    # Test queries showing different strengths
    test_cases = [
        {
            "query": "PRIMERGY TX1310",
            "description": "Exact keyword match - BM25 should excel",
            "expected": "BM25 finds exact product name"
        },
        {
            "query": "How do computers learn from data?",
            "description": "Semantic query - Vector should excel", 
            "expected": "Vector finds 'machine learning' despite different words"
        },
        {
            "query": "AI and machine learning",
            "description": "Mixed query - Hybrid should work best",
            "expected": "Both methods contribute to good results"
        },
        {
            "query": "server location building room",
            "description": "Multiple keywords - Testing hybrid fusion",
            "expected": "Finds asset document with location info"
        }
    ]
    
    for test in test_cases:
        print_header(f"Query: \"{test['query']}\"")
        print(f"📝 {test['description']}")
        print(f"💡 Expected: {test['expected']}")
        
        # BM25 Search
        bm25_results = bm25_retriever.retrieve(test['query'], top_k=3)
        print_results(bm25_results, "BM25 (Keyword)")
        
        # Vector Search
        vector_results = vector_retriever.retrieve(test['query'], top_k=3)
        print_results(vector_results, "Vector (Semantic)")
        
        # Hybrid Search
        hybrid_results = hybrid_retriever.retrieve(
            test['query'], 
            top_k=3, 
            method="weighted"
        )
        print_results(hybrid_results, "Hybrid (Combined)")
    
    # Show method comparison
    print_header("🔬 Fusion Method Comparison")
    
    query = "machine learning algorithms"
    print(f"\nQuery: \"{query}\"")
    print("\nComparing fusion methods:")
    
    for method in ["weighted", "rrf", "max"]:
        results = hybrid_retriever.retrieve(query, top_k=3, method=method)
        print(f"\n  📌 {method.upper()} Fusion:")
        for i, r in enumerate(results, 1):
            print(f"    [{i}] Score: {r.score:.3f} - {r.content[:60]}...")
    
    # Interactive mode
    print_header("🎮 Interactive Search")
    print("Type a query and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Query> ").strip()
        except EOFError:
            break
            
        if query.lower() in ['quit', 'exit', 'q', '']:
            print("Goodbye! 👋")
            break
        
        results = hybrid_retriever.retrieve(query, top_k=5, method="weighted")
        print_results(results, "Hybrid Search")
        print()


if __name__ == "__main__":
    main()
