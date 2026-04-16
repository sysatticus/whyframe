"""Attribution engine - trace decisions to their source."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import re


@dataclass
class Decision:
    """Represents a code decision."""
    id: str
    title: str
    description: str
    commits: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    pr_number: Optional[int] = None
    issue_numbers: list[int] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime | None = None


@dataclass
class AttributionResult:
    """Result of attribution query."""
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    commit_hashes: list[str] = field(default_factory=list)
    confidence: float = 0.0


class AttributionEngine:
    """Trace code decisions to their source."""

    def __init__(self, git_parser, embedding_pipeline):
        self.git_parser = git_parser
        self.embedding_pipeline = embedding_pipeline
        self._decision_cache: dict[str, Decision] = {}

    def attribute_decision(
        self,
        file_path: str,
        question: str,
    ) -> AttributionResult:
        """Attribute why a file/function was changed."""
        # Get file history
        commits = self.git_parser.get_file_history(file_path)
        
        if not commits:
            return AttributionResult(
                question=question,
                answer=f"No commit history found for {file_path}",
                sources=[],
                confidence=0.0,
            )
        
        # Extract PR and issue references from commit messages
        pr_pattern = re.compile(r"#(\d+)")
        issue_pattern = re.compile(r"(?:closes?|fixes?|resolves?)\s+#(\d+)", re.IGNORECASE)
        
        relevant_commits = []
        pr_numbers = set()
        issue_numbers = set()
        
        for commit in commits:
            # Check if commit message contains relevant keywords
            if any(kw in commit.message.lower() for kw in ["add", "implement", "fix", "refactor", "change", "update"]):
                relevant_commits.append(commit)
                
                # Extract PR numbers
                for match in pr_pattern.findall(commit.message):
                    pr_numbers.add(int(match))
                
                # Extract issue numbers
                for match in issue_pattern.findall(commit.message):
                    issue_numbers.add(int(match))
        
        # Build answer from most recent relevant commit
        if relevant_commits:
            latest = relevant_commits[0]
            
            sources = [
                {
                    "type": "commit",
                    "hash": latest.hash,
                    "message": latest.message,
                    "author": latest.author,
                    "date": latest.date.isoformat(),
                }
            ]
            
            answer = (
                f"Based on commit {latest.hash[:8]} by {latest.author}:\n\n"
                f"{latest.message}\n\n"
                f"This commit touched: {', '.join(latest.changed_files[:5])}"
            )
            
            if pr_numbers:
                answer += f"\n\nRelated PRs: {', '.join(f'#{n}' for n in list(pr_numbers)[:3])}"
            
            confidence = 0.8 if len(relevant_commits) > 1 else 0.6
        else:
            answer = f"Could not determine decision reason for {file_path}"
            sources = []
            confidence = 0.3
        
        return AttributionResult(
            question=question,
            answer=answer,
            sources=sources,
            commit_hashes=[c.hash for c in relevant_commits[:3]],
            confidence=confidence,
        )

    def attribute_function(self, function_name: str, code: str) -> AttributionResult:
        """Attribute why a function exists."""
        # Embed function and search for similar commits
        result = self.embedding_pipeline.embed_function(function_name, code)
        
        # For now, return a placeholder
        return AttributionResult(
            question=f"Why was function {function_name} added?",
            answer="Function attribution requires commit graph analysis. Coming in Phase 2.",
            sources=[],
            confidence=0.0,
        )

    def find_related_decisions(self, file_path: str, limit: int = 5) -> list[Decision]:
        """Find all decisions related to a file."""
        commits = self.git_parser.get_file_history(file_path)
        
        decisions = []
        for commit in commits[:limit]:
            decision = Decision(
                id=commit.hash,
                title=commit.message.split("\n")[0][:50],
                description=commit.message,
                commits=[commit.hash],
                files=commit.changed_files,
                created_at=commit.date,
            )
            decisions.append(decision)
        
        return decisions

    def calculate_confidence(self, commits: list, question_relevance: float) -> float:
        """Calculate confidence score for attribution."""
        if not commits:
            return 0.0
        
        # Factors:
        # - Number of commits (more = higher confidence)
        # - Question relevance (from embedding similarity)
        # - Time decay (older commits = lower confidence)
        
        commit_factor = min(len(commits) / 10, 1.0)  # Max out at 10 commits
        time_factor = 1.0  # Could add time decay
        
        confidence = (commit_factor * 0.5 + question_relevance * 0.5) * time_factor
        return min(confidence, 1.0)