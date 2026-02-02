"""
Verifier Module
Detects hallucinations by validating generated claims against retrieved context.
Uses LLM-based verification for better reasoning and reliability.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class Verifier:
    """
    Hallucination Detection using LLM "Self-Correction" logic.
    
    Instead of a local NLI model (which can have dependency issues), 
    this uses the LLM to perform 'Fact Verification'.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def extract_claims(self, text: str) -> List[str]:
        """Splits the answer into individual verifiable claims."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def verify(self, answer: str, context_chunks: List[Any]) -> Dict[str, Any]:
        """
        Validates the answer against the context using the LLM.
        """
        # Format context for the LLM
        context_text = "\n".join([f"- {c.content}" for c in context_chunks])
        
        system_prompt = (
            "You are a Fact-Checking Verifier. Your job is to compare an 'Answer' against 'Source Context' "
            "and identify any hallucinations or unsupported claims.\n\n"
            "Rules:\n"
            "1. Be extremely strict. If a claim is not directly supported by the context, flag it.\n"
            "2. Label each claim as: 'supported', 'contradicted', or 'unsupported' (neutral).\n"
            "3. Provide a brief reason for your judgment.\n"
            "Return a JSON object with 'is_hallucination_suspected' (bool), 'confidence_score' (0-1), "
            "and 'claim_verifications' (list of objects with 'claim', 'label', 'reason')."
        )
        
        prompt = f"Source Context:\n{context_text}\n\nAnswer to Verify:\n{answer}"
        
        logger.info("Verifying answer for potential hallucinations...")
        try:
            result = self.llm.generate_json(prompt, system_prompt=system_prompt)
            return {
                "is_hallucination_suspected": result.get("is_hallucination_suspected", True),
                "confidence_score": result.get("confidence_score", 0.0),
                "claim_verifications": result.get("claim_verifications", [])
            }
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "is_hallucination_suspected": True,
                "error": str(e),
                "confidence_score": 0.0,
                "claim_verifications": []
            }
