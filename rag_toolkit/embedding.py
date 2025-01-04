from abc import ABC, abstractmethod
import os
from typing import List, Optional, Any
import numpy as np
import requests
import voyageai
from dataclasses import dataclass


@dataclass
class EmbeddingResponse:
    """Standardized response from any embedding provider"""
    embeddings: List[List[float]]
    usage: dict
    raw_response: Any = None


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings for the given texts"""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings"""
        pass


class VoyageProvider(EmbeddingProvider):
    """Embedding provider using Voyage AI"""

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-2"):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key is required")
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        response = self.client.embed(texts, model=self.model)

        return EmbeddingResponse(
            embeddings=response.embeddings,
            usage={"total_tokens": sum(len(text.split()) for text in texts)},  # Estimation
            raw_response=response
        )

    def get_embedding_dim(self) -> int:
        # Voyage-2 embeddings are 1024-dimensional
        return 1024


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using Ollama"""

    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self._embedding_dim = None

        # Verify Ollama is running
        try:
            requests.get(f"{self.host}/api/health")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to Ollama. Ensure it's running and accessible."
            )

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        embeddings = []

        for text in texts:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            embeddings.append(result["embedding"])

            # Cache embedding dimensionality
            if self._embedding_dim is None:
                self._embedding_dim = len(result["embedding"])

        return EmbeddingResponse(
            embeddings=embeddings,
            usage={"total_tokens": sum(len(text.split()) for text in texts)},  # Estimation
            raw_response=None
        )

    def get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            # Generate a test embedding to determine dimensionality
            response = self.embed(["test"])
            self._embedding_dim = len(response.embeddings[0])
        return self._embedding_dim


class EmbeddingFactory:
    """Factory class to create embedding providers"""

    @staticmethod
    def create_provider(provider: str, **kwargs) -> EmbeddingProvider:
        providers = {
            "voyage": VoyageProvider,
            "ollama": OllamaEmbeddingProvider
        }

        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available providers: {list(providers.keys())}")

        return providers[provider](**kwargs)


class VectorSimilarity:
    """Utility class for vector similarity calculations"""

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1_array = np.array(v1)
        v2_array = np.array(v2)

        dot_product = np.dot(v1_array, v2_array)
        norm_v1 = np.linalg.norm(v1_array)
        norm_v2 = np.linalg.norm(v2_array)

        return dot_product / (norm_v1 * norm_v2)

    @staticmethod
    def batch_cosine_similarity(query: List[float], documents: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between a query and multiple documents"""
        query_array = np.array(query)
        doc_array = np.array(documents)

        # Normalize vectors
        query_norm = np.linalg.norm(query_array)
        doc_norms = np.linalg.norm(doc_array, axis=1)

        # Calculate similarities
        similarities = np.dot(doc_array, query_array) / (doc_norms * query_norm)
        return similarities.tolist()


# Example usage:
if __name__ == "__main__":
    # Using Voyage
    voyage_embed = EmbeddingFactory.create_provider("voyage")
    texts = ["Hello, world!", "Another text"]
    response = voyage_embed.embed(texts)
    print(f"Voyage embeddings shape: {len(response.embeddings)}x{len(response.embeddings[0])}")
    print(f"Token usage: {response.usage}")

    # Using Ollama
    try:
        ollama_embed = EmbeddingFactory.create_provider("ollama", model="llama2")
        response = ollama_embed.embed(texts)
        print(f"\nOllama embeddings shape: {len(response.embeddings)}x{len(response.embeddings[0])}")
        print(f"Estimated token usage: {response.usage}")

        # Calculate similarity between the two texts
        sim = VectorSimilarity.cosine_similarity(
            response.embeddings[0],
            response.embeddings[1]
        )
        print(f"Similarity between texts: {sim:.4f}")
    except ConnectionError as e:
        print(f"\nCouldn't connect to Ollama: {e}")