"""Search and retrieval - find relevant code and commits."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from whyframe.pipeline.embeddings import EmbeddingPipeline
from whyframe.core.git_parser import GitParser, Commit


@dataclass
class SearchResult:
    """Result from semantic search."""
    text: str
    score: float
    type: str  # "file", "commit", "function"
    source: str  # file path or commit hash


class SearchEngine:
    """Semantic search over indexed code."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline, git_parser: GitParser):
        self.embedding_pipeline = embedding_pipeline
        self.git_parser = git_parser
        self._file_index: dict[str, list[float]] = {}
        self._commit_index: dict[str, list[float]] = {}

    def index_file(self, file_path: str, content: str):
        """Add a file to the search index."""
        result = self.embedding_pipeline.embed_code(content)
        self._file_index[file_path] = result.vector

    def index_commit(self, commit_hash: str, message: str):
        """Add a commit to the search index."""
        result = self.embedding_pipeline.embed_code(message)
        self._commit_index[commit_hash] = result.vector

    def search(
        self,
        query: str,
        limit: int = 10,
        search_type: Optional[str] = None,  # "files", "commits", "all"
    ) -> list[SearchResult]:
        """Search indexed content."""
        # Generate query embedding
        query_result = self.embedding_pipeline.embed_code(query)
        query_vector = np.array(query_result.vector)
        
        results = []
        
        # Search files
        if search_type in (None, "files"):
            for file_path, vector in self._file_index.items():
                score = self._cosine_similarity(query_vector, np.array(vector))
                results.append(SearchResult(
                    text=file_path,
                    score=score,
                    type="file",
                    source=file_path,
                ))
        
        # Search commits
        if search_type in (None, "commits"):
            for commit_hash, vector in self._commit_index.items():
                score = self._cosine_similarity(query_vector, np.array(vector))
                results.append(SearchResult(
                    text=commit_hash,
                    score=score,
                    type="commit",
                    source=commit_hash,
                ))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def search_files(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search only files."""
        return self.search(query, limit, "files")

    def search_commits(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search only commits."""
        return self.search(query, limit, "commits")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_index_stats(self) -> dict:
        """Get index statistics."""
        return {
            "files_indexed": len(self._file_index),
            "commits_indexed": len(self._commit_index),
            "embedding_dim": len(next(iter(self._file_index.values()), [])),
        }

    def clear_index(self):
        """Clear all indexed content."""
        self._file_index.clear()
        self._commit_index.clear()