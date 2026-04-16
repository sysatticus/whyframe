"""Git history parser - extracts commits, relationships, and metadata."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess


@dataclass
class Commit:
    """Represents a git commit."""
    hash: str
    author: str
    author_email: str
    date: datetime
    message: str
    parent_hashes: list[str]
    changed_files: list[str]
    diff: Optional[str] = None


class GitParser:
    """Parse git history and extract commit metadata."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")

    def get_commit(self, hash: str) -> Commit:
        """Get a single commit by hash."""
        # Get commit info
        info = subprocess.run(
            ["git", "show", "--format=%H%n%an%n%ae%n%aI%n%s", "-s", hash],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        lines = info.stdout.strip().split("\n")
        
        # Get parent hashes
        parents = subprocess.run(
            ["git", "rev-list", "--parents", "-n", "1", hash],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        parent_hashes = parents.stdout.strip().split()[1:] if parents.stdout.strip() else []
        
        # Get changed files
        files = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", hash],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        changed_files = [f for f in files.stdout.strip().split("\n") if f]
        
        return Commit(
            hash=lines[0],
            author=lines[1],
            author_email=lines[2],
            date=datetime.fromisoformat(lines[3]),
            message=lines[4] if len(lines) > 4 else "",
            parent_hashes=parent_hashes,
            changed_files=changed_files,
        )

    def get_all_commits(self, max_count: Optional[int] = None) -> list[Commit]:
        """Get all commits in the repository."""
        cmd = ["git", "log", "--format=%H"]
        if max_count:
            cmd.extend(["-n", str(max_count)])
        
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        hashes = result.stdout.strip().split("\n")
        
        commits = []
        for h in hashes:
            if h:
                try:
                    commits.append(self.get_commit(h))
                except Exception:
                    continue
        return commits

    def get_file_history(self, file_path: str) -> list[Commit]:
        """Get all commits that touched a specific file."""
        result = subprocess.run(
            ["git", "log", "--format=%H", "--", file_path],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        hashes = result.stdout.strip().split("\n")
        
        commits = []
        for h in hashes:
            if h:
                try:
                    commits.append(self.get_commit(h))
                except Exception:
                    continue
        return commits

    def get_commit_range(self, start_hash: str, end_hash: str) -> list[Commit]:
        """Get commits in a range (inclusive)."""
        result = subprocess.run(
            ["git", "log", "--format=%H", f"{start_hash}..{end_hash}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        hashes = result.stdout.strip().split("\n")
        
        commits = []
        for h in hashes:
            if h:
                try:
                    commits.append(self.get_commit(h))
                except Exception:
                    continue
        return commits

    def get_branch_name(self) -> str:
        """Get current branch name."""
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def get_all_branches(self) -> list[str]:
        """Get all branch names."""
        result = subprocess.run(
            ["git", "branch", "-a"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return [b.strip() for b in result.stdout.strip().split("\n") if b]