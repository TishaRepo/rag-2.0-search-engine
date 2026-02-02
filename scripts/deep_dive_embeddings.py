"""
NGSE: Deep Dive into Embeddings
This script demonstrates how local text embeddings work under the hood.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a, b):
    """Calculates the angle (similarity) between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # 1. Load the Model (The "Brain")
    # This downloads a ~100MB model file to your computer.
    print("🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Define some sentences to compare
    sentences = [
        "The computer is fast.",              # Original
        "That laptop has great speed.",       # Semantic match (different words, same meaning)
        "I like eating pepperoni pizza.",     # Unrelated
        "machine learning is a subset of AI"  # Technical
    ]
    
    # 3. Generate Embeddings (The Math)
    print("\n🔢 Converting sentences to 384-dimensional vectors...")
    embeddings = model.encode(sentences)
    
    for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
        print(f"\nSentence {i+1}: '{sentence}'")
        # Show only the first 5 numbers of the 384 numbers in the vector
        print(f"Vector (start): {emb[:5]} ... (total length: {len(emb)})")

    # 4. Calculate Similarity
    # Let's compare Sentence 1 to the others
    base_emb = embeddings[0]
    
    print("\n📊 Comparing Similarity to Sentence 1 ('The computer is fast.'):")
    print("-" * 60)
    
    for i in range(1, len(sentences)):
        score = cosine_similarity(base_emb, embeddings[i])
        
        # A score closer to 1.0 means high similarity
        if score > 0.6:
            tag = "✅ STRONG MATCH"
        elif score > 0.3:
            tag = "⚠️ WEAK MATCH"
        else:
            tag = "❌ NO MATCH"
            
        print(f"vs Sentence {i+1}: {score:.4f} | {tag}")

if __name__ == "__main__":
    main()
