"""
LLM Client Utility
Production-ready unified interface for LLM providers (Groq, OpenAI, Anthropic).
Includes retry logic, timeout handling, and structured error responses.
"""

import os
import json
import logging
from typing import Dict, Optional, Any

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from src.config import LLM_PROVIDER, LLM_MODELS, OPENAI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class LLMClient:
    """
    Production-ready client for interacting with LLM providers.
    
    Features:
    - Automatic retry with exponential backoff
    - Configurable timeouts
    - Structured error handling
    - Support for multiple providers (OpenAI, Groq, Anthropic)
    
    Usage:
        client = LLMClient()
        response = client.generate("What is AI?")
        json_response = client.generate_json("List 3 facts about AI")
    """
    
    # Default configuration
    DEFAULT_TIMEOUT = 60  # seconds
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TEMPERATURE = 0.0
    
    def __init__(
        self, 
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider (openai, groq, anthropic)
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODELS.get(self.provider)
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize client based on provider
        self._init_client()
        
        logger.info(f"LLM Client initialized: provider={self.provider}, model={self.model}")
    
    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise LLMError("OPENAI_API_KEY not found in environment. Please set it in .env file.")
            self.client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                timeout=self.timeout
            )
            
        elif self.provider == "groq":
            if not GROQ_API_KEY:
                raise LLMError("GROQ_API_KEY not found in environment. Please set it in .env file.")
            self.client = openai.OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                timeout=self.timeout
            )
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise LLMError("ANTHROPIC_API_KEY not found in environment. Please set it in .env file.")
            try:
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=ANTHROPIC_API_KEY,
                    timeout=self.timeout
                )
            except ImportError:
                raise LLMError("Anthropic provider requires 'anthropic' package. Install with: pip install anthropic")
        else:
            raise LLMError(f"Unsupported LLM provider: {self.provider}. Supported: openai, groq, anthropic")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "You are a helpful assistant.", 
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a text response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response (optional)
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: If generation fails after retries
        """
        try:
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            
            if not content:
                logger.warning("Empty response from LLM")
                return ""
            
            return content
            
        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed for {self.provider}: {e}")
            raise LLMError(f"Invalid API key for {self.provider}. Please check your credentials.")
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit for {self.provider}: {e}")
            raise  # Will be retried
            
        except openai.APIConnectionError as e:
            logger.warning(f"Connection error for {self.provider}: {e}")
            raise  # Will be retried
            
        except openai.APIError as e:
            logger.error(f"API error from {self.provider}: {e}")
            raise LLMError(f"API error from {self.provider}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error generating response: {e}")
            raise LLMError(f"Failed to generate response: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    def generate_json(
        self, 
        prompt: str, 
        system_prompt: str = "You are a helpful assistant.", 
        temperature: float = DEFAULT_TEMPERATURE
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            LLMError: If generation or parsing fails
        """
        try:
            # Enhance system prompt for JSON output
            json_system_prompt = system_prompt + "\n\nIMPORTANT: ALWAYS return your response as a valid JSON object. Do not include any text outside the JSON."
            
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            
            # Use structured output if supported
            if self.provider in ["openai", "groq"]:
                request_params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            
            if not content:
                logger.warning("Empty JSON response from LLM")
                return {}
            
            # Parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # Try to extract JSON from the response
                logger.warning(f"Failed to parse JSON response, attempting extraction: {e}")
                return self._extract_json(content)
                
        except openai.RateLimitError:
            raise  # Will be retried
            
        except openai.APIConnectionError:
            raise  # Will be retried
            
        except LLMError:
            raise
            
        except Exception as e:
            logger.error(f"Error generating JSON response: {e}")
            return {"error": str(e), "raw_error": True}
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Attempt to extract JSON from text that may contain extra content.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON as dictionary
        """
        import re
        
        # Try to find JSON object pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return error
        logger.error(f"Could not extract JSON from response: {text[:200]}...")
        return {"error": "Failed to parse JSON from LLM response", "raw_response": text[:500]}
    
    def is_healthy(self) -> bool:
        """
        Check if the LLM connection is healthy.
        
        Returns:
            True if connection is working
        """
        try:
            response = self.generate("Reply with 'ok'", max_tokens=5)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
