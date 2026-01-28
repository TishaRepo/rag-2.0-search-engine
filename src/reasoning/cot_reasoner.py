"""
CoT Reasoner Module
Implements Chain-of-Thought reasoning for synthesizing answers from multiple retrieved documents.
"""

import logging
from typing import List, Dict, Any, Optional
from src.utils.llm_client import LLMClient
from src.retrieval.bm25_retriever import RetrievalResult

logger = logging.getLogger(__name__)

class CoTReasoner:
    """
    Synthesizes answers using multi-hop Chain-of-Thought reasoning.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def reason(self, query: str, context_chunks: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Processes query using the provided context chunks and returns a reasoned answer.
        """
        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk.metadata.get("title") or chunk.metadata.get("filename") or "Unknown"
            context_text += f"Source [{i+1}] ({source}):\n{chunk.content}\n\n"

        system_prompt = (
            "You are a Reasoning Engine for a Next-Gen Search Engine. "
            "Your task is to answer user queries using the provided search results. "
            "Follow these steps:\n"
            "1. Analyze the context and identify relevant facts.\n"
            "2. If the user prompt is complex, break down your reasoning step-by-step (Chain-of-Thought).\n"
            "3. Synthesize a final answer based ONLY on the provided context.\n"
            "4. Mention the sources you used.\n"
            "Return a JSON object with 'reasoning_steps' (list) and 'final_answer' (string)."
        )
        
        prompt = f"User Query: {query}\n\nContext:\n{context_text}"
        
        logger.info("Executing reasoning pipeline...")
        result = self.llm.generate_json(prompt, system_prompt=system_prompt)
        
        return {
            "query": query,
            "reasoning_steps": result.get("reasoning_steps", []),
            "final_answer": result.get("final_answer", "I couldn't find enough information to answer that."),
            "source_count": len(context_chunks)
        }
