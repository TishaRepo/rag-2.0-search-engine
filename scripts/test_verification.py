"""
Test Verification (Phase 5) - LLM Powered
Demonstrates hallucination detection using LLM-based fact checking.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Env Vars
from dotenv import load_dotenv
load_dotenv()

from src.config import GROQ_API_KEY, OPENAI_API_KEY
from src.retrieval.bm25_retriever import RetrievalResult
from src.reasoning import Verifier

def main():
    print("=" * 70)
    print("🛡️  Phase 5: Verifier Model Demo (LLM Edition)")
    print("=" * 70)

    # Check for API Keys
    if not GROQ_API_KEY and not OPENAI_API_KEY:
        print("\n❌ Error: No API keys found! Please set GROQ_API_KEY or OPENAI_API_KEY in a .env file.")
        return

    # 1. Initialize Verifier
    print("\n⏳ Initializing LLM Verifier...")
    verifier = Verifier()

    # 2. Setup Context
    context = [
        RetrievalResult(
            chunk_id="1",
            content="The PRIMERGY TX1310 server is located in Room 208 of the High Throughput Building.",
            score=1.0,
            metadata={"filename": "asset-report.txt"}
        )
    ]

    # 3. Test Case A: Accurate Answer
    good_answer = "The PRIMERGY TX1310 server is in Room 208."
    print(f"\n✅ Testing Accurate Answer: '{good_answer}'")
    v1 = verifier.verify(good_answer, context)
    print(f"Confidence Score: {v1['confidence_score']}")
    print(f"Hallucination suspected? {v1['is_hallucination_suspected']}")

    # 4. Test Case B: Hallucinated Answer
    bad_answer = "The PRIMERGY TX1310 server is located in a basement in London."
    print(f"\n❌ Testing Hallucinated Answer: '{bad_answer}'")
    v2 = verifier.verify(bad_answer, context)
    print(f"Confidence Score: {v2['confidence_score']}")
    print(f"Hallucination suspected? {v2['is_hallucination_suspected']}")
    
    for c in v2.get('claim_verifications', []):
        print(f"  Claim: {c['claim']}")
        print(f"  Result: {c['label']}")
        print(f"  Reason: {c['reason']}")

    print("\n" + "=" * 70)
    print("✅ Verification test complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
