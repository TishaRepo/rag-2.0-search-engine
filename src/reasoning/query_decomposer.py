"""
Query Decomposer Module
Breaks down complex user queries into simpler, searchable sub-queries.
"""

import logging
from typing import List, Dict, Optional
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class QueryDecomposer:
    """
    Decomposes complex queries into atomic sub-queries.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def decompose(self, query: str) -> List[str]:
        """
        Decomposes a query into sub-queries.
        """
        system_prompt = (
            "You are a Query Decomposer for a RAG search engine. "
            "Your goal is to take a complex user query and break it down into several simple, atomic sub-queries "
            "that can be answered independently using a search engine. "
            "Example: 'What is the capital of France and what is its population?' -> ['What is the capital of France?', 'What is the population of Paris?'] "
            "Return a JSON object with a 'sub_queries' list."
        )
        
        prompt = f"Decompose the following complex query into simple sub-queries: '{query}'"
        
        logger.info(f"Decomposing query: {query}")
        result = self.llm.generate_json(prompt, system_prompt=system_prompt)
        
        sub_queries = result.get("sub_queries", [query])
        logger.info(f"Generated {len(sub_queries)} sub-queries: {sub_queries}")
        
        return sub_queries
