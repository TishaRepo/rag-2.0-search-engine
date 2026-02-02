import requests
import json

# Replace with your actual Hugging Face Direct URL
# You can find this in Space Settings -> Embed this Space -> Direct URL
SPACE_URL = "https://tishamadame-rag-search-engine.hf.space"

def ingest_test_data():
    # Note: We flattened the metadata into the main dict to be safe with all vector stores
    documents = [
        {
            "id": "ml-001",
            "content": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
            "source": "wikipedia",
            "topic": "AI"
        },
        {
            "id": "ml-002",
            "content": "A dual-socket rack server like the PRIMERGY RX2540 M4 is designed for high-performance computing in data centers.",
            "device": "server",
            "source": "manual"
        },
        {
            "id": "rag-001",
            "content": "RAG 2.0 combines retrieval with reasoning and verification to eliminate hallucinations in LLM responses.",
            "source": "tech-blog"
        }
    ]

    print(f"📡 Sending {len(documents)} documents to {SPACE_URL}/ingest...")
    
    try:
        response = requests.post(
            f"{SPACE_URL}/ingest",
            json={
                "documents": documents,
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ Success! Data ingested.")
            result = response.json()
            print(f"Documents processed: {result.get('documents_processed')}")
            print(f"Chunks created: {result.get('chunks_created')}")
        else:
            print(f"❌ Failed! Status Code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    ingest_test_data()
