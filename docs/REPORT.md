# Project Report
1. **Objective & scope** - Insurance knowledge assistant for claims handlers, policy servicing, customer support; RAG + semantic search + MCP on AWS Bedrock.
2. **Architecture** - see `docs/architecture.md`.
3. **Corpus** - 8 synthetic documents in `corpus/` (policy wording, coverage guide, claims procedure, claims manual, FAQ, regulatory, 2 role-restricted). Deliberate conflict: reporting window 14 days (POL-001) vs 30 days (FAQ-001).
4. **Implementation** - ingest.py (extraction, section-aware chunking 1200/150, Titan embeddings), vectorstore.py (role-filtered cosine), rag.py (condense follow-ups, score gate, cited generation, conflict/insufficient statuses), mcp_server.py, app.py.
5. **Safeguards** - answers only from passages; citations mandatory (uncited output discarded); no coverage/claim decisions; role filter before retrieval; Bedrock errors handled.
6. **Evaluation** - run `python eval/run_eval.py`; paste `eval/report.md` here.
7. **Known limitations** - see README and report.
8. **Prerequisites** - see README.
