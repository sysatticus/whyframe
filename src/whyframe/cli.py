"""CLI for Whyframe."""
import sys
from pathlib import Path
import argparse

from whyframe.core.config import Config
from whyframe.pipeline.indexer import Indexer


def cmd_index(args):
    """Index a repository."""
    repo_path = Path(args.repo)
    
    if not repo_path.exists():
        print(f"Error: Repository not found: {repo_path}")
        sys.exit(1)
    
    if not (repo_path / ".git").exists():
        print(f"Error: Not a git repository: {repo_path}")
        sys.exit(1)
    
    # Load config if provided
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    config.repo_path = repo_path
    if args.embedding_model:
        config.embedding.model = args.embedding_model
    if args.max_commits:
        config.git.max_commit_history = args.max_commits
    
    # Run indexer
    indexer = Indexer(config)
    
    print(f"Indexing repository: {repo_path}")
    result = indexer.index_repo(
        max_files=args.max_files,
        max_commits=args.max_commits,
        show_progress=not args.quiet,
    )
    
    print(f"\nIndexing complete!")
    print(f"  Files indexed: {result.indexed_files}")
    print(f"  Commits indexed: {result.indexed_commits}")
    print(f"  Duration: {result.duration_seconds:.2f}s")


def cmd_ask(args):
    """Ask a question about the codebase."""
    print(f"Question: {args.question}")
    print("\nNote: Ask functionality not yet implemented.")
    print("Coming in Phase 2-3.")


def cmd_setup(args):
    """Run the interactive setup wizard."""
    from whyframe.setup import run_setup
    run_setup()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="whyframe",
        description="Ask your repo why this exists — codebase reasoning via embeddings and graph",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("repo", help="Path to repository")
    index_parser.add_argument("--max-files", type=int, help="Maximum files to index")
    index_parser.add_argument("--max-commits", type=int, help="Maximum commits to index")
    index_parser.add_argument("--embedding-model", type=str, help="Embedding model to use")
    index_parser.add_argument("--config", type=str, help="Path to config file")
    index_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    index_parser.set_defaults(func=cmd_index)
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the codebase")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.set_defaults(func=cmd_ask)
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run interactive setup wizard")
    setup_parser.set_defaults(func=cmd_setup)
    
    # Onboard command (alias for setup)
    onboard_parser = subparsers.add_parser("onboard", help="Run interactive setup wizard")
    onboard_parser.set_defaults(func=cmd_setup)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()