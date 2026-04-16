"""Configuration management for Whyframe."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    provider: str = "openai"  # openai, local, anthropic
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    cache_dir: Path | None = None


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    provider: str = "pinecone"  # pinecone, weaviate, pgvector
    api_key: str = ""
    environment: str = "us-west1"
    index_name: str = "whyframe"
    metric: str = "cosine"


@dataclass
class GraphDBConfig:
    """Configuration for graph database."""
    provider: str = "neo4j"  # neo4j, networkx (in-memory)
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""


@dataclass
class GitConfig:
    """Configuration for git parsing."""
    max_commit_history: int = 10000
    include_diff: bool = False
    ignored_paths: list[str] = field(default_factory=lambda: [".git", "__pycache__", "node_modules", "*.pyc"])


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
        with open(path) as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if "embedding" in data:
            for k, v in data["embedding"].items():
                setattr(config.embedding, k, v)
        if "vector_db" in data:
            for k, v in data["vector_db"].items():
                setattr(config.vector_db, k, v)
        if "graph_db" in data:
            for k, v in data["graph_db"].items():
                setattr(config.graph_db, k, v)
        if "git" in data:
            for k, v in data["git"].items():
                setattr(config.git, k, v)
        
        return config

    def to_file(self, path: Path | str):
        """Save config to YAML file."""
        data = {
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size,
            },
            "vector_db": {
                "provider": self.vector_db.provider,
                "environment": self.vector_db.environment,
                "index_name": self.vector_db.index_name,
                "metric": self.vector_db.metric,
            },
            "graph_db": {
                "provider": self.graph_db.provider,
                "uri": self.graph_db.uri,
            },
            "git": {
                "max_commit_history": self.git.max_commit_history,
                "include_diff": self.git.include_diff,
                "ignored_paths": self.git.ignored_paths,
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)