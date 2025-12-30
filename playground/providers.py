"""
LLM Provider Adapters for Playground
Supports: HuggingFace, Groq
Dynamically fetches available models from each provider's API
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging
import requests

logger = logging.getLogger(__name__)

# Cache for fetched models (to avoid repeated API calls)
_model_cache = {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Generate a response from the LLM."""
        pass
    
    @staticmethod
    @abstractmethod
    def fetch_available_models(api_key: str) -> list[dict]:
        """Fetch available models from the provider's API."""
        pass


class GroqProvider(LLMProvider):
    """Groq API provider adapter."""
    
    # Fallback models if API call fails
    FALLBACK_MODELS = [
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile", "context_window": 32768},
        {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant", "context_window": 8192},
        {"id": "llama-3.2-3b-preview", "name": "Llama 3.2 3B Preview", "context_window": 8192},
        {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "context_window": 32768},
        {"id": "gemma2-9b-it", "name": "Gemma 2 9B IT", "context_window": 8192},
    ]
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._init_client()
    
    def _init_client(self):
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Groq client: {e}")
    
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.client:
            raise RuntimeError("Groq client not initialized")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content.strip()
    
    @staticmethod
    def fetch_available_models(api_key: str) -> list[dict]:
        """Fetch available models from Groq API."""
        cache_key = f"groq_{api_key[:8]}"
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        
        try:
            # Groq uses OpenAI-compatible API - GET /models
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("data", []):
                # Filter to only include chat/completion models
                model_id = model.get("id", "")
                # Skip whisper (audio) and other non-chat models
                if "whisper" in model_id.lower() or "tts" in model_id.lower():
                    continue
                    
                models.append({
                    "id": model_id,
                    "name": model_id.replace("-", " ").title(),
                    "context_window": model.get("context_window", 8192),
                    "owned_by": model.get("owned_by", "groq")
                })
            
            # Sort by name
            models.sort(key=lambda x: x["name"])
            _model_cache[cache_key] = models
            logger.info(f"Fetched {len(models)} models from Groq API")
            return models
            
        except Exception as e:
            logger.warning(f"Failed to fetch Groq models: {e}. Using fallback list.")
            return GroqProvider.FALLBACK_MODELS


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider adapter."""
    
    # Fallback models if API call fails
    FALLBACK_MODELS = [
        {"id": "meta-llama/Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B Instruct"},
        {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B Instruct"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B Instruct v0.3"},
        {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi 3 Mini 4K Instruct"},
        {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B IT"},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B Instruct"},
    ]
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._init_client()
    
    def _init_client(self):
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
            logger.info(f"HuggingFace client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("huggingface_hub package not installed. Run: pip install huggingface_hub")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize HuggingFace client: {e}")
    
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.client:
            raise RuntimeError("HuggingFace client not initialized")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content.strip()
    
    @staticmethod
    def fetch_available_models(api_key: str) -> list[dict]:
        """Fetch available text-generation models from HuggingFace Hub."""
        cache_key = f"hf_{api_key[:8]}"
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        
        try:
            # Use HuggingFace Hub API to list models available for inference
            # Filter by text-generation-inference and conversational pipelines
            response = requests.get(
                "https://huggingface.co/api/models",
                params={
                    "pipeline_tag": "text-generation",
                    "filter": "conversational",
                    "inference": "warm",  # Only models with warm inference endpoints
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": 50
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data:
                model_id = model.get("modelId", model.get("id", ""))
                # Only include models that support inference
                if model.get("inference") == "warm" or model.get("pipeline_tag") == "text-generation":
                    models.append({
                        "id": model_id,
                        "name": model_id.split("/")[-1].replace("-", " "),
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0)
                    })
            
            if not models:
                # If no warm models, get popular text-generation models
                response = requests.get(
                    "https://huggingface.co/api/models",
                    params={
                        "pipeline_tag": "text-generation",
                        "sort": "downloads",
                        "direction": "-1",
                        "limit": 30
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                
                for model in data:
                    model_id = model.get("modelId", model.get("id", ""))
                    # Filter to well-known instruct models
                    if any(x in model_id.lower() for x in ["instruct", "chat", "it"]):
                        models.append({
                            "id": model_id,
                            "name": model_id.split("/")[-1].replace("-", " "),
                            "downloads": model.get("downloads", 0),
                            "likes": model.get("likes", 0)
                        })
            
            _model_cache[cache_key] = models
            logger.info(f"Fetched {len(models)} models from HuggingFace Hub")
            return models if models else HuggingFaceProvider.FALLBACK_MODELS
            
        except Exception as e:
            logger.warning(f"Failed to fetch HuggingFace models: {e}. Using fallback list.")
            return HuggingFaceProvider.FALLBACK_MODELS


def create_provider(
    provider_name: str,
    api_key: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> LLMProvider:
    """
    Factory function to create an LLM provider.
    
    Args:
        provider_name: 'groq' or 'huggingface'
        api_key: API key for the provider
        model: Model name (optional, uses default if not specified)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        LLMProvider instance
    """
    provider_name = provider_name.lower()
    
    if provider_name == "groq":
        return GroqProvider(
            api_key=api_key,
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider_name in ["huggingface", "hf"]:
        return HuggingFaceProvider(
            api_key=api_key,
            model=model or "meta-llama/Llama-3.2-3B-Instruct",
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: groq, huggingface")


def get_provider_info() -> dict:
    """Return information about available providers."""
    return {
        "groq": {
            "name": "Groq",
            "default_model": "llama-3.3-70b-versatile",
            "api_key_url": "https://console.groq.com/keys",
            "supports_dynamic_models": True
        },
        "huggingface": {
            "name": "HuggingFace",
            "default_model": "meta-llama/Llama-3.2-3B-Instruct",
            "api_key_url": "https://huggingface.co/settings/tokens",
            "supports_dynamic_models": True
        }
    }


def fetch_models_for_provider(provider_name: str, api_key: str) -> list[dict]:
    """Fetch available models for a specific provider."""
    provider_name = provider_name.lower()
    
    if provider_name == "groq":
        return GroqProvider.fetch_available_models(api_key)
    elif provider_name in ["huggingface", "hf"]:
        return HuggingFaceProvider.fetch_available_models(api_key)
    else:
        return []

