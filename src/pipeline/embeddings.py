"""Embedding pipeline - convert code to vectors."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib

from openai import OpenAI


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    vector: list[float]
    model: str
    token_count: Optional[int] = None


class EmbeddingPipeline:
    """Generate embeddings for code and text."""

    SUPPORTED_MODELS = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        # Qwen
        "qwen/qwen3-embedding-8b": 1024,
        # Mistral
        "mistralai/codestral-embed-2505": 1024,
        # open source
        "thenlper/gte-large": 1024,
        "intfloat/multilingual-e5-large": 1024,
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        self.client = OpenAI(
            api_key=api_key or "dummy",  # Will be overridden if base_url provided
            base_url=base_url,
        )
        
        if base_url and not api_key:
            # For OpenAI-compatible APIs, we need a key but it can be dummy
            self.client.api_key = "not-needed"
        
        self.model = model
        self.dimension = self.SUPPORTED_MODELS.get(model, 1536)
        self.base_url = base_url
        self._cache: dict[str, list[float]] = {}

    def embed_code(self, code: str, use_cache: bool = True) -> EmbeddingResult:
        """Generate embedding for code snippet."""
        # Check cache
        if use_cache:
            cache_key = self._make_cache_key(code)
            if cache_key in self._cache:
                return EmbeddingResult(
                    text=code,
                    vector=self._cache[cache_key],
                    model=self.model,
                )

        # Generate embedding
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=code,
            )
            
            vector = response.data[0].embedding
            
            # Cache result
            if use_cache:
                cache_key = self._make_cache_key(code)
                self._cache[cache_key] = vector
            
            return EmbeddingResult(
                text=code,
                vector=vector,
                model=self.model,
                token_count=response.usage.total_tokens if response.usage else None,
            )
        except Exception as e:
            # Return zero vector on error
            return EmbeddingResult(
                text=code,
                vector=[0.0] * self.dimension,
                model=self.model,
            )

    def embed_batch(self, texts: list[str], use_cache: bool = True) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        # Filter cached
        results = []
        to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._make_cache_key(text)
                if cache_key in self._cache:
                    results.append(EmbeddingResult(
                        text=text,
                        vector=self._cache[cache_key],
                        model=self.model,
                    ))
                    continue
            to_embed.append(text)
            indices.append(i)
        
        # Batch embed remaining
        if to_embed:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=to_embed,
                )
                
                for i, data in enumerate(response.data):
                    text = to_embed[i]
                    result = EmbeddingResult(
                        text=text,
                        vector=data.embedding,
                        model=self.model,
                        token_count=response.usage.total_tokens if response.usage else None,
                    )
                    results.append(result)
                    
                    if use_cache:
                        cache_key = self._make_cache_key(text)
                        self._cache[cache_key] = data.embedding
            except Exception as e:
                # Return zero vectors on error
                for text in to_embed:
                    results.append(EmbeddingResult(
                        text=text,
                        vector=[0.0] * self.dimension,
                        model=self.model,
                    ))
        
        return results

    def embed_file(self, file_path: Path) -> EmbeddingResult:
        """Generate embedding for a file's contents."""
        try:
            content = file_path.read_text(errors="ignore")
            return self.embed_code(content)
        except Exception as e:
            return EmbeddingResult(
                text=f"Error reading {file_path}: {e}",
                vector=[0.0] * self.dimension,
                model=self.model,
            )

    def embed_function(self, function_name: str, code: str) -> EmbeddingResult:
        """Generate embedding for a function with context."""
        # Add function name as context
        context = f"Function: {function_name}\n\nCode:\n{code}"
        return self.embed_code(context)

    def _make_cache_key(self, text: str) -> str:
        """Create a cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all supported models."""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_dimension(cls, model: str) -> int:
        """Get embedding dimension for a model."""
        return cls.SUPPORTED_MODELS.get(model, 1536)