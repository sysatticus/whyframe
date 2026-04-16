"""Configuration management for Whyframe."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    cache_dir: Path | None = None
    base_url: str = ""
    api_key: str = ""


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""

    provider: str = "pinecone"
    api_key: str = ""
    environment: str = "us-west1"
    index_name: str = "whyframe"
    metric: str = "cosine"


@dataclass
class GraphDBConfig:
    """Configuration for graph database."""

    provider: str = "neo4j"
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""


@dataclass
class GitConfig:
    """Configuration for git parsing."""

    max_commit_history: int = 10000
    include_diff: bool = False
    ignored_paths: list[str] = field(
        default_factory=lambda: [".git", "__pycache__", "node_modules", "*.pyc"]
    )


@dataclass
class Config:
    """Main configuration."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = field(default_factory=GraphDBConfig)
    git: GitConfig = field(default_factory=GitConfig)
    repo_path: Path | None = None

    @classmethod
    def from_file(cls, path: Path | str) -> "Config":
        """Load config from YAML file."""
        with open(path) as file_handle:
            data = yaml.safe_load(file_handle) or {}

        config = cls()
        if "embedding" in data:
            for key, value in data["embedding"].items():
                setattr(config.embedding, key, value)
        if "vector_db" in data:
            for key, value in data["vector_db"].items():
                setattr(config.vector_db, key, value)
        if "graph_db" in data:
            for key, value in data["graph_db"].items():
                setattr(config.graph_db, key, value)
        if "git" in data:
            for key, value in data["git"].items():
                setattr(config.git, key, value)

        return config

    def to_file(self, path: Path | str) -> None:
        """Save config to YAML file."""
        data = {
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size,
                "base_url": self.embedding.base_url,
                "api_key": self.embedding.api_key,
            },
            "vector_db": {
                "provider": self.vector_db.provider,
                "api_key": self.vector_db.api_key,
                "environment": self.vector_db.environment,
                "index_name": self.vector_db.index_name,
                "metric": self.vector_db.metric,
            },
            "graph_db": {
                "provider": self.graph_db.provider,
                "uri": self.graph_db.uri,
                "user": self.graph_db.user,
                "password": self.graph_db.password,
            },
            "git": {
                "max_commit_history": self.git.max_commit_history,
                "include_diff": self.git.include_diff,
                "ignored_paths": self.git.ignored_paths,
            },
        }
        with open(path, "w") as file_handle:
            yaml.dump(data, file_handle, default_flow_style=False)
