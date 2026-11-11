import re
from access import validate_role
from config import INDEX_DIR, TOP_K, MIN_SCORE
from vectorstore import VectorStore
from errors import BedrockError

SYSTEM = """You are an internal Insurance Policy and Claims Knowledge Assistant for insurance employees.
Answer ONLY from the numbered context passages provided. Rules:
1. Cite every factual statement with passage tags like [S1] or [S2][S3].
2. If the passages do not contain the answer, reply starting with exactly "INSUFFICIENT_EVIDENCE:" and say what is missing. Never guess.
3. If passages disagree, start with exactly "CONFLICT:" then state each version with its citation and say it must be confirmed with the document owner or a senior adjuster.
4. Never invent policy terms, limits, or dates. Never decide whether a specific claim is covered, approved or denied; describe what the documents say and refer the decision to an authorized claims adjudicator.
5. Be concise. Use short bullet points for lists of documents, exclusions or steps."""

INSUFFICIENT_MSG = ("INSUFFICIENT_EVIDENCE: The approved documents available to your role do not contain "
                    "enough information to answer this question. Please consult the relevant document owner or a senior colleague.")

class Assistant:
    def __init__(self, store=None, embed_fn=None, llm_fn=None):
        self.store = store or VectorStore.load(INDEX_DIR)
        if embed_fn is None or llm_fn is None:
            import bedrock_client as bc
            embed_fn, llm_fn = embed_fn or bc.embed, llm_fn or bc.converse
        self.embed, self.llm = embed_fn, llm_fn

    def search(self, query: str, role: str, k: int = TOP_K):
        validate_role(role)
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        return self.store.search(self.embed(query.strip()), role, k)

    def _standalone(self, question, history):
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        return self.llm("Rewrite the final question as a standalone insurance question using the conversation for context. Return only the question.",
                        f"Conversation:\n{convo}\n\nFinal question: {question}", 150).strip() or question

    def ask(self, question: str, role: str, history=None):
        validate_role(role)
        if not question or not question.strip():
            return {"status": "error", "answer": "Please enter a question.", "sources": []}
        try:
            standalone = self._standalone(question.strip(), history)
            hits = [h for h in self.store.search(self.embed(standalone), role, TOP_K) if h["score"] >= MIN_SCORE]
            if not hits:
                return {"status": "insufficient_evidence", "answer": INSUFFICIENT_MSG, "sources": [], "query": standalone}
            ctx = "\n\n".join(f"[S{i}] ({h['doc_id']} - {h['title']} - {h['section']})\n{h['text']}"
                              for i, h in enumerate(hits, 1))
            answer = self.llm(SYSTEM, f"Context passages:\n{ctx}\n\nQuestion: {standalone}")
        except BedrockError as e:
            return {"status": "error", "answer": f"The service could not complete the request: {e}", "sources": []}
        cited = {int(n) for n in re.findall(r"\[S(\d+)\]", answer)}
        sources = [{"ref": f"S{i}", "doc_id": h["doc_id"], "title": h["title"], "section": h["section"],
                    "score": round(h["score"], 3), "text": h["text"], "cited": i in cited}
                   for i, h in enumerate(hits, 1)]
        up = answer.upper()
        if up.startswith("INSUFFICIENT_EVIDENCE"):
            status = "insufficient_evidence"
        elif not cited:
            status, answer = "insufficient_evidence", INSUFFICIENT_MSG  # ungrounded output is discarded
        elif up.startswith("CONFLICT"):
            status = "conflict"
        else:
            status = "ok"
        return {"status": status, "answer": answer, "sources": sources, "query": standalone}
