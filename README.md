# Insurance Policy and Claims Knowledge Assistant

AWS Bedrock (Titan Embeddings v2 + Claude) - RAG - semantic search - MCP server - Streamlit UI.

## Prerequisites (external, required)
1. AWS account with credentials configured (`aws configure` or `AWS_PROFILE`); no keys in code.
2. Bedrock model access enabled in your region for **Titan Text Embeddings V2** and a **Claude** model.
3. IAM permission `bedrock:InvokeModel` for both models.
4. Python 3.10+.

## Setup
```
cd insurance-assistant
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set AWS_REGION and LLM_MODEL_ID (Claude model/inference-profile ID)
python ingest.py            # builds index/ from corpus/
```
## Run
- Web UI: `streamlit run app.py`
- MCP server (stdio): `MCP_USER_ROLE=claims_handler python mcp_server.py`
  Client config example (e.g. Claude Desktop `mcpServers`):
  `{"insurance-kb":{"command":"python","args":["/abs/path/mcp_server.py"],"env":{"MCP_USER_ROLE":"claims_handler"}}}`
  Tools: `search_documents(query, top_k)`, `ask_question(question, history)`.

## Test
- Offline logic tests (no AWS; stub models): `python tests/test_offline.py`
- Live evaluation (needs AWS): `python eval/run_eval.py` -> writes `eval/report.md`

## Corpus format
Files in `corpus/` (.md/.txt/.pdf) start with front matter: `doc_id`, `title`, `allowed_roles: [role, ...]`. PDFs need a sidecar `<file>.pdf.meta` holding those keys. Re-run `python ingest.py` after changes.

## Demo scenarios (UI)
1 Coverage: "Is a burst pipe covered?" | 2 Exclusions: "What does the policy exclude?" | 3 Documents: "Which documents are required to file a claim?" | 4 Process: "What is the claims processing procedure?" | 5 Follow-up: ask "Is sewer backup covered?" then "What is the limit?" | 6 Insufficient: "What is the policy on cyber insurance for small businesses?" | Conflict: "How long to report a loss?" (POL-001 says 14 days, FAQ-001 says 30) | Access: ask the SIU question as customer_support vs claims_handler.

## Design notes / limits
- Vector search is a persisted local cosine index (numpy) to avoid adding infrastructure; `VectorStore` is the single swap point for a managed vector service.
- Role is chosen in the UI (simulated sign-in) or by MCP env var; production needs a real identity source.
- Tune `MIN_SCORE` after your first eval run.
- Status of verification: offline logic tests pass; Bedrock, Streamlit and MCP paths were NOT executed here (no network/AWS in the build environment).
