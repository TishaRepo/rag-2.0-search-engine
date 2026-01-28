"""
Test Reasoning Pipeline (Phase 4)
Integrates Hybrid Retrieval, Reranking, and Reasoning.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INDICES_DIR, GROQ_API_KEY, OPENAI_API_KEY
from src.retrieval import HybridRetriever, Reranker
from src.reasoning import QueryDecomposer, CoTReasoner

def main():
    print("=" * 70)
    print("🚀 Phase 4: Reasoning Engine Demo")
    print("=" * 70)

    # Check for API Keys
    if not GROQ_API_KEY and not OPENAI_API_KEY:
        print("\n❌ Error: No API keys found! Please set GROQ_API_KEY or OPENAI_API_KEY in a .env file.")
        print("See .env.example for a template.")
        return

    # 1. Initialize Components
    print("\n⏳ Loading components...")
    hybrid_retriever = HybridRetriever(
        bm25_index_path=INDICES_DIR / "bm25_index.pkl",
        chroma_persist_dir=str(INDICES_DIR / "chroma"),
        alpha=0.5
    )
    reranker = Reranker()
    decomposer = QueryDecomposer()
    reasoner = CoTReasoner()
    print("✅ Components loaded.")

    # 2. Define a Complex Query
    query = "Compare Artificial Intelligence and Machine Learning. Which one uses neural networks?"
    print(f"\n❓ Original Query: '{query}'")

    # 3. Query Decomposition
    print("\n✂️  Decomposing query...")
    sub_queries = decomposer.decompose(query)
    for i, sq in enumerate(sub_queries, 1):
        print(f"  [{i}] {sq}")

    # 4. Retrieval & Reranking for each sub-query
    print("\n🔍 Retrieving and Reranking context...")
    all_context_chunks = []
    seen_chunk_ids = set()

    for sq in sub_queries:
        # Hybrid Search
        results = hybrid_retriever.retrieve(sq, top_k=5)
        # Rerank
        reranked = reranker.rerank(sq, results, top_k=3)
        
        for r in reranked:
            if r.chunk_id not in seen_chunk_ids:
                all_context_chunks.append(r)
                seen_chunk_ids.add(r.chunk_id)

    print(f"✅ Gathered {len(all_context_chunks)} unique context chunks.")

    # 5. Reasoning
    print("\n🧠 Reasoning over context...")
    result = reasoner.reason(query, all_context_chunks)

    print("\n📝 Reasoning Steps:")
    for i, step in enumerate(result['reasoning_steps'], 1):
        print(f"  {i}. {step}")

    print("\n🏆 Final Answer:")
    print("-" * 70)
    print(result['final_answer'])
    print("-" * 70)

    print("\n" + "=" * 70)
    print("✅ Reasoning test complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
