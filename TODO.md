# Whyframe — Project Roadmap

## Phase 0: Foundation (Week 1-2)
- [ ] Finalize architecture doc (data flow, components)
- [ ] Choose vector DB (Pinecone / Weaviate / pgvector)
- [ ] Choose embedding model (OpenAI text-embedding-3-small vs local)
- [ ] Set up dev environment (Python 3.11+, dependencies)
- [ ] Create project structure (src/, tests/, docs/)
- [ ] Write unit tests for core modules

## Phase 1: Core Pipeline (Week 3-5)
- [ ] Build git history parser (commit → relationship mapping)
- [ ] Build file/function parser (AST-based, tree-sitter)
- [ ] Build embedding pipeline (code → vectors)
- [ ] Build vector storage (index, search, retrieve)
- [ ] Build graph DB schema (Neo4j or PostgreSQL + relational)
- [ ] Connect embeddings to graph relationships
- [ ] Basic "ask" endpoint (semantic search + graph traversal)

## Phase 2: Attribution Engine (Week 6-7)
- [ ] Map commits → files → functions
- [ ] Extract PR context (GitHub API integration)
- [ ] Extract issue references
- [ ] Build "source of truth" attribution (which commit owns what)
- [ ] Handle multi-file decisions (refactors touching many files)
- [ ] Add confidence scoring for answers

## Phase 3: UI / UX (Week 8-10)
- [ ] CLI tool (ask questions, get answers)
- [ ] VS Code extension prototype (hover → "why?")
- [ ] Web dashboard (visualize decision graph)
- [ ] Slack/Discord bot integration
- [ ] RAG-ready API for AI agents

## Phase 4: Intelligence (Week 11-13)
- [ ] Decision decay scoring (age + churn rate → risk)
- [ ] Rollback confidence calculator
- [ ] Multi-repo support (enterprise)
- [ ] Custom context (team docs, RFCs)
- [ ] Fine-tuned model for codebase-specific answers

## Phase 5: Scale & Ops (Week 14+)
- [ ] Incremental indexing (only new commits)
- [ ] Branch-aware indexing (main vs feature branches)
- [ ] Cloud deployment (Docker, Kubernetes)
- [ ] Multi-tenant auth (enterprise SSO)
- [ ] Usage analytics (who's asking what)
- [ ] Pricing tiers (free / team / enterprise)

---

## Tech Stack
- **Language:** Python 3.11+
- **Embedding:** OpenAI text-embedding-3-small (or local)
- **Vector DB:** Pinecone / Weaviate / pgvector
- **Graph:** Neo4j / PostgreSQL + NetworkX
- **Parsing:** tree-sitter, AST
- **Git:** GitHub API, pygit2
- **Frontend:** FastAPI + React
- **Deployment:** Docker, Railway / AWS

---

## Key Milestones
1. **MVP** — Index a repo, ask a question, get a traced answer (Week 7)
2. **Beta** — CLI + web UI working (Week 10)
3. **Launch** — Public beta, early users (Week 14)
4. **Scale** — Enterprise features (Week 18+)

---

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Attribution hallucination | Ground answers in git metadata, not LLM |
| Slow indexing | Incremental updates, parallel processing |
| Cost (embeddings) | Cache aggressively, use smaller models for bulk |
| Multi-repo scaling | Separate indexes per repo, federated query |

---

## Priority Order
1. Git parsing + embedding pipeline
2. Basic search + retrieval
3. Attribution (the hard part)
4. CLI for testing
5. VS Code extension
6. Web UI
7. Enterprise features

---

## Questions to Answer Early
- [ ] Open-source version vs SaaS only?
- [ ] Self-hosted option?
- [ ] Which verticals to target first? (AI teams, legacy codebases, startups)
- [ ] Pricing model?