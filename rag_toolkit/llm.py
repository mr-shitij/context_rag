from abc import ABC, abstractmethod
import os
from typing import Optional, Dict, Any
import anthropic
import requests
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    usage: Dict[str, Any]
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> LLMResponse:
        """Generate text from the LLM"""
        pass


class AnthropicProvider(LLMProvider):
    """Claude provider using Anthropic's API with support for context caching"""

    # Token costs in USD per 1K tokens (as of 2024)
    COST_PER_1K_TOKENS = {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 0.75},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0}
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-3-haiku-20240307",
                 use_context_cache: bool = False):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.use_context_cache = use_context_cache

        # Get token costs for the model
        self.token_costs = self.COST_PER_1K_TOKENS.get(
            model,
            {"input": 0.25, "output": 0.75}  # Default to Haiku prices
        )

    def generate(self,
                 prompt: str,
                 context: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> LLMResponse:
        """Generate text with optional context caching"""

        if not self.use_context_cache or context is None:
            # Prepare params, omitting None values
            params = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": prompt if context is None else f"{context}\n\n{prompt}"
                }]
            }
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature

            response = self.client.messages.create(**params)

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (
                    (input_tokens * self.token_costs["input"] / 1000) +
                    (output_tokens * self.token_costs["output"] / 1000)
            )

            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost, 6)
                },
                raw_response=response
            )

        # Use context caching
        # Prepare params for cached request
        params = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": context,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature

        params["extra_headers"] = {"anthropic-beta": "prompt-caching-2024-07-31"}

        response = self.client.beta.prompt_caching.messages.create(**params)

        # Calculate costs with cache consideration
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_creation_tokens = response.usage.cache_creation_input_tokens
        cache_read_tokens = response.usage.cache_read_input_tokens

        # Cached tokens get 90% discount
        cost = (
                (input_tokens * self.token_costs["input"] / 1000) +
                (output_tokens * self.token_costs["output"] / 1000) +
                (cache_creation_tokens * self.token_costs["input"] / 1000) +
                (cache_read_tokens * self.token_costs["input"] * 0.1 / 1000)  # 90% discount
        )

        return LLMResponse(
            content=response.content[0].text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cost_usd": round(cost, 6)
            },
            raw_response=response
        )


class OllamaProvider(LLMProvider):
    """Local LLM provider using Ollama"""

    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

        # Verify Ollama is running
        try:
            requests.get(f"{self.host}/api/health")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to Ollama. Ensure it's running and accessible."
            )

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> LLMResponse:
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # Add options only if specified
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload
        )
        response.raise_for_status()
        result = response.json()

        # Get token counts from Ollama response
        response_text = result["response"]
        input_tokens = result.get("prompt_eval_count", len(prompt.split()))
        output_tokens = result.get("eval_count", len(response_text.split()))

        return LLMResponse(
            content=response_text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": 0.0  # Local model, no API costs
            },
            raw_response=result
        )


class LLMFactory:
    """Factory class to create LLM providers"""

    @staticmethod
    def create_provider(provider: str, **kwargs) -> LLMProvider:
        providers = {
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider
        }

        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available providers: {list(providers.keys())}")

        return providers[provider](**kwargs)


# Example usage:
if __name__ == "__main__":
    # Basic usage
    anthropic_llm = LLMFactory.create_provider("anthropic")
    response = anthropic_llm.generate("Tell me a short joke")
    print(f"Basic response: {response.content}")
    print(f"Usage: {response.usage}")

    # With specific parameters
    response = anthropic_llm.generate(
        prompt="Write a long story",
        max_tokens=2000,
        temperature=0.7
    )

    # Using context caching
    cached_llm = AnthropicProvider(use_context_cache=True)
    context = "The Theory of Relativity was developed by Einstein in 1905."
    response = cached_llm.generate(
        prompt="When was it developed?",
        context=context
    )
    print(f"\nCached response: {response.content}")
    print(f"Usage with caching: {response.usage}")

    # Using Ollama
    try:
        ollama_llm = LLMFactory.create_provider("ollama", model="llama2")
        response = ollama_llm.generate("Tell me a joke")
        print(f"\nOllama response: {response.content}")
        print(f"Usage: {response.usage}")
    except ConnectionError as e:
        print(f"\nCouldn't connect to Ollama: {e}")