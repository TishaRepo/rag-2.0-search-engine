"""
Document Loader Module
Handles loading documents from various sources and formats.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    
    content: str
    metadata: Dict = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate document ID if not provided."""
        if self.doc_id is None:
            # Create hash from content for unique ID
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        """Create document from dictionary."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id")
        )


class DocumentLoader:
    """
    Load documents from various sources and formats.
    
    Supported formats:
    - Text files (.txt)
    - Markdown files (.md)
    - JSON files (.json)
    - Wikipedia dumps
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}
    
    def __init__(self, source_dir: Optional[Path] = None):
        """
        Initialize document loader.
        
        Args:
            source_dir: Default directory to load documents from
        """
        self.source_dir = Path(source_dir) if source_dir else None
        self.loaded_docs: List[Document] = []
    
    def load_file(self, file_path: Path) -> Optional[Document]:
        """
        Load a single file as a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object or None if loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file format: {extension}")
            return None
        
        try:
            if extension == ".json":
                return self._load_json(file_path)
            else:
                return self._load_text(file_path)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def _load_text(self, file_path: Path) -> Document:
        """Load a text or markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "format": file_path.suffix,
            "loaded_at": datetime.now().isoformat()
        }
        
        return Document(content=content, metadata=metadata)
    
    def _load_json(self, file_path: Path) -> Document:
        """Load a JSON file (expects 'content' field or full text)."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, str):
            content = data
            metadata = {}
        elif isinstance(data, dict):
            content = data.get("content", data.get("text", json.dumps(data)))
            metadata = {k: v for k, v in data.items() if k not in ["content", "text"]}
        else:
            content = json.dumps(data)
            metadata = {}
        
        metadata.update({
            "source": str(file_path),
            "filename": file_path.name,
            "format": ".json",
            "loaded_at": datetime.now().isoformat()
        })
        
        return Document(content=content, metadata=metadata)
    
    def load_directory(
        self, 
        directory: Optional[Path] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory to load from (uses source_dir if None)
            recursive: Whether to search subdirectories
            extensions: List of extensions to include (e.g., [".txt", ".md"])
            
        Returns:
            List of loaded documents
        """
        directory = Path(directory) if directory else self.source_dir
        
        if not directory or not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        extensions = set(extensions) if extensions else self.SUPPORTED_EXTENSIONS
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                doc = self.load_file(file_path)
                if doc:
                    documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        self.loaded_docs.extend(documents)
        return documents
    
    def load_wikipedia_articles(self, articles_data: List[Dict]) -> List[Document]:
        """
        Load Wikipedia articles from a list of article dictionaries.
        
        Args:
            articles_data: List of dicts with 'title' and 'text' keys
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for article in articles_data:
            title = article.get("title", "Unknown")
            text = article.get("text", article.get("content", ""))
            
            if not text.strip():
                continue
            
            metadata = {
                "title": title,
                "source": "wikipedia",
                "url": article.get("url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                "loaded_at": datetime.now().isoformat()
            }
            
            doc = Document(content=text, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} Wikipedia articles")
        self.loaded_docs.extend(documents)
        return documents
    
    def create_sample_corpus(self, output_dir: Path) -> List[Document]:
        """
        Create a sample corpus for testing.
        Downloads or generates sample documents.
        
        Args:
            output_dir: Directory to save sample documents
            
        Returns:
            List of created documents
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample content for testing (can be replaced with actual Wikipedia download)
        sample_articles = [
            {
                "title": "Artificial Intelligence",
                "text": """Artificial intelligence (AI) is intelligence demonstrated by machines, 
                as opposed to natural intelligence displayed by animals including humans. 
                AI research has been defined as the field of study of intelligent agents, 
                which refers to any system that perceives its environment and takes actions 
                that maximize its chance of achieving its goals.

                The term "artificial intelligence" had previously been used to describe machines 
                that mimic and display "human" cognitive skills that are associated with the human mind, 
                such as "learning" and "problem-solving". This definition has since been rejected 
                by major AI researchers who now describe AI in terms of rationality and acting rationally.

                Machine learning is a subset of AI that enables systems to learn and improve from 
                experience without being explicitly programmed. Deep learning, a further subset, 
                uses neural networks with multiple layers to analyze data."""
            },
            {
                "title": "Machine Learning",
                "text": """Machine learning (ML) is a type of artificial intelligence (AI) that 
                allows software applications to become more accurate at predicting outcomes 
                without being explicitly programmed to do so. Machine learning algorithms use 
                historical data as input to predict new output values.

                There are four basic approaches to machine learning: supervised learning, 
                unsupervised learning, semi-supervised learning, and reinforcement learning. 
                The type of algorithm used depends on the type of data and the desired output.

                Supervised learning uses labeled datasets to train algorithms to classify data 
                or predict outcomes accurately. Unsupervised learning uses machine learning 
                algorithms to analyze and cluster unlabeled datasets. Reinforcement learning 
                is a type of machine learning where an agent learns to make decisions by 
                performing actions in an environment to maximize cumulative reward."""
            },
            {
                "title": "Natural Language Processing",
                "text": """Natural language processing (NLP) is a subfield of linguistics, 
                computer science, and artificial intelligence concerned with the interactions 
                between computers and human language, in particular how to program computers 
                to process and analyze large amounts of natural language data.

                Challenges in natural language processing frequently involve speech recognition, 
                natural language understanding, and natural language generation. NLP techniques 
                are used in various applications such as machine translation, sentiment analysis, 
                chatbots, and text summarization.

                Recent advances in deep learning have significantly improved NLP capabilities. 
                Transformer models like BERT and GPT have achieved state-of-the-art results 
                on many NLP benchmarks. These models are pre-trained on large text corpora 
                and can be fine-tuned for specific tasks."""
            },
            {
                "title": "Retrieval Augmented Generation",
                "text": """Retrieval-Augmented Generation (RAG) is an AI framework that combines 
                information retrieval with text generation. It enhances large language models (LLMs) 
                by allowing them to reference external knowledge sources before generating responses.

                In a RAG system, when a query is received, relevant documents are first retrieved 
                from a knowledge base using semantic search or keyword matching. These documents 
                are then provided as context to the language model, which generates a response 
                grounded in the retrieved information.

                RAG helps address several limitations of standard LLMs including: hallucinations 
                (generating false information), outdated knowledge, and lack of source attribution. 
                By grounding responses in retrieved documents, RAG systems can provide more accurate 
                and verifiable answers.

                Key components of a RAG system include: a document store, an embedding model for 
                semantic search, a retriever to fetch relevant documents, and a language model 
                for answer generation. Advanced RAG systems may also include re-ranking, 
                query decomposition, and verification modules."""
            },
            {
                "title": "Vector Databases",
                "text": """A vector database is a type of database that stores data as high-dimensional 
                vectors, which are mathematical representations of features or attributes. 
                These vectors enable similarity searches, where the database can find items 
                that are most similar to a query vector.

                Vector databases are essential for modern AI applications, particularly those 
                involving natural language processing and recommendation systems. They enable 
                semantic search, where queries return results based on meaning rather than 
                just keyword matching.

                Popular vector databases include Pinecone, Weaviate, Milvus, and ChromaDB. 
                These systems use approximate nearest neighbor (ANN) algorithms to efficiently 
                search through millions or billions of vectors. Common ANN algorithms include 
                HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index).

                The choice of embedding model significantly impacts search quality. Models like 
                OpenAI's text-embedding-ada-002, Sentence-BERT, and Cohere's embeddings are 
                commonly used to convert text into vectors for storage and retrieval."""
            }
        ]
        
        # Save sample articles as JSON
        documents = []
        for article in sample_articles:
            filename = article["title"].lower().replace(" ", "_") + ".json"
            file_path = output_dir / filename
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=2)
            
            doc = Document(
                content=article["text"],
                metadata={
                    "title": article["title"],
                    "source": str(file_path),
                    "format": ".json"
                }
            )
            documents.append(doc)
        
        logger.info(f"Created sample corpus with {len(documents)} documents in {output_dir}")
        self.loaded_docs.extend(documents)
        return documents
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded documents."""
        if not self.loaded_docs:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc.content) for doc in self.loaded_docs)
        avg_chars = total_chars // len(self.loaded_docs)
        
        return {
            "total_documents": len(self.loaded_docs),
            "total_characters": total_chars,
            "average_characters": avg_chars,
            "sources": list(set(doc.metadata.get("source", "unknown") for doc in self.loaded_docs))
        }


if __name__ == "__main__":
    # Test the document loader
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import RAW_DATA_DIR
    
    loader = DocumentLoader()
    
    # Create and load sample corpus
    docs = loader.create_sample_corpus(RAW_DATA_DIR)
    
    print(f"\n📊 Corpus Statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n📝 Sample Document:")
    if docs:
        print(f"  Title: {docs[0].metadata.get('title')}")
        print(f"  Content preview: {docs[0].content[:200]}...")
