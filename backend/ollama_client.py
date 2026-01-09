"""Ollama API client for making LLM requests."""
import httpx
from typing import List, Dict, Any, Optional
from .config import OLLAMA_BASE_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    ollama_url: str = OLLAMA_BASE_URL
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via Ollama API.
    
    Args:
        model: Ollama model identifier (e.g., "llama3:latest")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        ollama_url: Base URL for Ollama API
    
    Returns:
        Response dict with 'content', or None if failed
    """
    # Convert messages to a single prompt for Ollama
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                'content': data.get('response', '')
            }
    
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    ollama_url: str = OLLAMA_BASE_URL
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.
    
    Args:
        models: List of Ollama model identifiers
        messages: List of message dicts to send to each model
        ollama_url: Base URL for Ollama API
    
    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio
    
    # Create tasks for all models
    tasks = [query_model(model, messages, ollama_url=ollama_url) for model in models]
    
    # Wait for all to complete
    responses = await asyncio.gather(*tasks)
    
    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
