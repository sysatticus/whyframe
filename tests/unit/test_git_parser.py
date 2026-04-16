"""Tests for git parser module."""
import pytest
from pathlib import Path
from datetime import datetime
from whyframe.core.git_parser import GitParser, Commit


def test_git_parser_init_with_valid_repo(tmp_path):
    """Test GitParser initialization with a valid git repo."""
    # Create a temp git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    
    # Create a file and commit
    (tmp_path / "test.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True)
    
    # Test parser
    parser = GitParser(tmp_path)
    assert parser.repo_path == tmp_path


def test_git_parser_init_with_invalid_repo():
    """Test GitParser initialization with invalid repo raises error."""
    with pytest.raises(ValueError):
        GitParser("/nonexistent/path")


def test_commit_dataclass():
    """Test Commit dataclass fields."""
    commit = Commit(
        hash="abc123",
        author="Test User",
        author_email="test@test.com",
        date=datetime.now(),
        message="Test commit",
        parent_hashes=["parent1", "parent2"],
        changed_files=["file1.py", "file2.py"],
    )
    
    assert commit.hash == "abc123"
    assert commit.author == "Test User"
    assert len(commit.parent_hashes) == 2
    assert len(commit.changed_files) == 2


# Import subprocess for tests
import subprocess