"""
Factory module for instantiating the appropriate Vision-Language Model service.
"""

import logging
from config.settings import settings
from models.vlm_base import VLMService

logger = logging.getLogger(__name__)

def get_vlm_service() -> VLMService:
    """
    Returns the appropriate VLM service based on the configured provider in settings.
    
    Dynamically imports the requested provider to avoid loading unused SDKs 
    (e.g., if you are using Groq, you don't need to load Anthropic's heavy library).
    """
    provider = settings.vlm_provider
    logger.debug(f"Instantiating VLM service for provider: {provider}")

    if provider == "anthropic":
        from models.vlm_anthropic import VLMAnthropic
        return VLMAnthropic()
        
    elif provider == "openai":
        from models.vlm_openai import VLMOpenAI
        return VLMOpenAI()
        
    elif provider == "groq":
        # Groq uses an OpenAI-compatible API, so we reuse the OpenAI class
        # but inject Groq's specific endpoints and keys.
        from models.vlm_openai import VLMOpenAI
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in configuration.")
            
        logger.info("Routing Groq provider through OpenAI-compatible client.")
        return VLMOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    
    elif provider == "openrouter":
        # OpenRouter uses an OpenAI-compatible API.  VLMOpenAI.__init__ will
        # auto-detect the OpenRouter key and set the correct base_url / headers.
        from models.vlm_openai import VLMOpenAI
        logger.info("Routing OpenRouter provider through OpenAI-compatible client.")
        return VLMOpenAI()
        
    elif provider == "gemini":
        from models.vlm_gemini import VLMGemini
        return VLMGemini()
        
    elif provider == "qwen":
        from models.vlm_qwen import VLMQwen
        return VLMQwen()
        
    else:
        # Fallback safeguard (though Pydantic validation in settings.py should catch this first)
        raise ValueError(f"Unsupported VLM provider: {provider}")