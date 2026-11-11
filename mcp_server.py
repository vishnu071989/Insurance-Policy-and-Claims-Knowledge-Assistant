"""MCP server (stdio). Role is fixed per server process via MCP_USER_ROLE, never by tool arguments."""
import os
from mcp.server.fastmcp import FastMCP
from access import validate_role
from rag import Assistant

ROLE = validate_role(os.getenv("MCP_USER_ROLE", "customer_support"))
mcp = FastMCP("insurance-knowledge-assistant")
_assistant = None
def _a():
    global _assistant
    _assistant = _assistant or Assistant()
    return _assistant

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> list:
    """Semantic search over approved insurance documents accessible to the configured role.
    Returns passages with doc_id, title, section and similarity score."""
    try:
        hits = _a().search(query, ROLE, max(1, min(top_k, 10)))
    except Exception as e:
        return [{"error": str(e)}]
    return [{k: h[k] for k in ("doc_id", "title", "section", "score", "text")} for h in hits]

@mcp.tool()
def ask_question(question: str, history: list[dict] | None = None) -> dict:
    """Evidence-grounded answer with source references. history: optional prior turns
    as [{"role": "user"|"assistant", "content": "..."}] for follow-up questions.
    status is ok | conflict | insufficient_evidence | error."""
    try:
        return _a().ask(question, ROLE, history)
    except Exception as e:
        return {"status": "error", "answer": str(e), "sources": []}

if __name__ == "__main__":
    mcp.run()
