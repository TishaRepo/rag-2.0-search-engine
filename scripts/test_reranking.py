"""
Test Reranking (Phase 3)
Demonstrates how the Cross-Encoder improves the accuracy of retrieval results.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INDICES_DIR
from src.retrieval import HybridRetriever, Reranker

def print_results(results, title: str):
    print(f"\n{title}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r.score:.3f} | {r.retrieval_method}")
        print(f"    Content: {r.content[:120]}...")

def main():
    print("=" * 70)
    print("🚀 Phase 3: Re-Ranking Demo")
    print("=" * 70)

    # 1. Initialize Hybrid Retriever and Reranker
    print("\n⏳ Loading components...")
    hybrid_retriever = HybridRetriever(
        bm25_index_path=INDICES_DIR / "bm25_index.pkl",
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        alpha=0.5
    )
    reranker = Reranker() # Uses default ms-marco-MiniLM model
    print("✅ Components loaded.")

    # 2. Define a query
    query = "Where exactly is the PRIMERGY server located in the building?"
    
    # 3. Step 1: Hybrid Retrieval (Phase 2)
    # Get top 10 from hybrid to show how reranker changes the order
    print(f"\n🔍 Query: '{query}'")
    hybrid_results = hybrid_retriever.retrieve(query, top_k=10)
    print_results(hybrid_results, "📊 HYBRID RETRIEVAL RESULTS (Phase 2)")

    # 4. Step 2: Reranking (Phase 3)
    # We take the hybrid results and pass them to the reranker
    print("\n✨ Applying Reranker...")
    reranked_results = reranker.rerank(query, hybrid_results, top_k=5)
    print_results(reranked_results, "🏆 RERANKED RESULTS (Phase 3)")

    print("\n" + "=" * 70)
    print("✅ Reranking test complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
