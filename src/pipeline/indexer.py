"""Indexer - orchestrates git parsing and embedding generation."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from whyframe.core.git_parser import GitParser, Commit
from whyframe.core.config import Config
from whyframe.pipeline.embeddings import EmbeddingPipeline


@dataclass
class IndexedFile:
    """Represents an indexed file."""
    path: str
    content: str
    embedding: list[float]
    last_modified: datetime
    commit_hash: str


@dataclass
class IndexedCommit:
    """Represents an indexed commit."""
    hash: str
    message: str
    author: str
    author_email: str
    date: datetime
    changed_files: list[str]
    embedding: list[float] | None = None


@dataclass
class IndexResult:
    """Result of indexing operation."""
    repo_path: str
    indexed_files: int
    indexed_commits: int
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class Indexer:
    """Orchestrate indexing of a repository."""

    def __init__(self, config: Config):
        self.config = config
        self.git_parser = GitParser(config.repo_path)
        self.embedding_pipeline = EmbeddingPipeline(
            api_key=config.embedding.api_key or None,
            base_url=config.embedding.base_url or None,
            model=config.embedding.model,
            dimension=config.embedding.dimension,
        )
        self._indexed_files: list[IndexedFile] = []
        self._indexed_commits: list[IndexedCommit] = []

    def index_repo(
        self,
        max_files: int | None = None,
        max_commits: int | None = None,
        show_progress: bool = True,
    ) -> IndexResult:
        """Index all files and commits in the repository."""
        start_time = datetime.now()
        
        # Index files
        repo_path = Path(self.config.repo_path)
        all_files = list(repo_path.rglob("*"))
        py_files = [f for f in all_files if f.is_file() and self._should_index(f)]
        
        if show_progress:
            print(f"Indexing {len(py_files)} files...")
        
        for file_path in tqdm(py_files[:max_files] if max_files else py_files, disable=not show_progress):
            try:
                result = self.embedding_pipeline.embed_file(file_path)
                indexed_file = IndexedFile(
                    path=str(file_path.relative_to(repo_path)),
                    content=file_path.read_text(errors="ignore"),
                    embedding=result.vector,
                    last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                    commit_hash="",
                )
                self._indexed_files.append(indexed_file)
            except Exception as e:
                continue
        
        # Index commits
        if show_progress:
            print(f"Indexing commits...")
        
        commits = self.git_parser.get_all_commits(max_count=max_commits)
        
        for commit in tqdm(commits, disable=not show_progress):
            try:
                # Embed commit message
                msg_embedding = self.embedding_pipeline.embed_code(commit.message)
                
                indexed_commit = IndexedCommit(
                    hash=commit.hash,
                    message=commit.message,
                    author=commit.author,
                    author_email=commit.author_email,
                    date=commit.date,
                    changed_files=commit.changed_files,
                    embedding=msg_embedding.vector,
                )
                self._indexed_commits.append(indexed_commit)
            except Exception as e:
                continue
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return IndexResult(
            repo_path=str(repo_path),
            indexed_files=len(self._indexed_files),
            indexed_commits=len(self._indexed_commits),
            duration_seconds=duration,
        )

    def index_incremental(self, since_commit: str) -> IndexResult:
        """Index only new commits since a given commit."""
        start_time = datetime.now()
        
        commits = self.git_parser.get_commit_range(since_commit, "HEAD")
        
        for commit in tqdm(commits):
            try:
                msg_embedding = self.embedding_pipeline.embed_code(commit.message)
                
                indexed_commit = IndexedCommit(
                    hash=commit.hash,
                    message=commit.message,
                    author=commit.author,
                    author_email=commit.author_email,
                    date=commit.date,
                    changed_files=commit.changed_files,
                    embedding=msg_embedding.vector,
                )
                self._indexed_commits.append(indexed_commit)
            except Exception as e:
                continue
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return IndexResult(
            repo_path=str(self.config.repo_path),
            indexed_files=0,
            indexed_commits=len(commits),
            duration_seconds=duration,
        )

    def _should_index(self, file_path: Path) -> bool:
        """Check if file should be indexed."""
        # Skip ignored paths
        for ignored in self.config.git.ignored_paths:
            if ignored.startswith("*"):
                if file_path.name.endswith(ignored[1:]):
                    return False
            elif ignored in str(file_path):
                return False
        
        # Only index code files
        code_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"}
        return file_path.suffix in code_extensions

    def get_indexed_files(self) -> list[IndexedFile]:
        """Get all indexed files."""
        return self._indexed_files

    def get_indexed_commits(self) -> list[IndexedCommit]:
        """Get all indexed commits."""
        return self._indexed_commits