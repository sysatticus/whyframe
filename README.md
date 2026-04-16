# Whyframe

> Ask your repo why this exists — codebase reasoning via embeddings and graph.

## Quick Start

```bash
# Install
pip install -e .

# Index a repository
whyframe index /path/to/repo

# Ask a question
whyframe ask "Why was this function added?"
```

## Whyframe

Whyframe helps developers understand the reasoning behind code decisions by:

1. **Indexing** - Parse git history, extract relationships, create embeddings
2. **Graphing** - Build decision graphs from commits, PRs, and code relationships
3. **Answering** - Semantic search + graph traversal to explain "why"

## Architecture

- `src/pipeline/` - Embedding and indexing pipeline
- `src/attribution/` - Decision attribution engine
- `src/graph/` - Graph database integration
- `src/api/` - REST API and CLI

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .
```

## License

MIT