"""
LLM Client Utility
Provides a unified interface for different LLM providers (Groq, OpenAI, Anthropic).
"""

import os
import logging
from typing import List, Dict, Optional, Any

import openai
from src.config import LLM_PROVIDER, LLM_MODELS, OPENAI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with LLM providers.
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODELS.get(self.provider)
        
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment")
            # Groq uses the OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            # Anthropic client would need 'anthropic' package
            # For now, let's assume we use OpenAI or Groq as they are more common in initial setups
            # If Anthropic is needed, 'pip install anthropic' and additional logic is required
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.0) -> str:
        """
        Generate a response from the LLM.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from {self.provider}: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.0) -> Dict:
        """
        Generate a JSON response from the LLM.
        Note: This is a simple wrapper. Some models support 'response_format={"type": "json_object"}'
        """
        # For Llama-3 on Groq or GPT-4, we can specify JSON format
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt + " ALWAYS return response as a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating JSON response: {e}")
            # Fallback: try to find JSON in the text if it failed
            return {"error": str(e)}
