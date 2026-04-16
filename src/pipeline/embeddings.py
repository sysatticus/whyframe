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

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension
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
            response = self.client.embeddings.create(
                model=self.model,
                input=to_embed,
            )
            
            for i, data in enumerate(response.data):
                text = to_embed[indices.index(i)] if indices.count(i) > 0 else to_embed[i]
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