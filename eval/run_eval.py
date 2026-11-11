"""Runs against live Bedrock. Writes eval/report.md with REAL measured results."""
import json, sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import Assistant
from config import TOP_K

def main():
    a = Assistant()
    qs = json.loads((Path(__file__).parent / "questions.json").read_text())
    rows, ret, acc, grd, con, safe = [], [], [], [], [], []
    for q in qs:
        hits = a.search(q["question"], q["role"], TOP_K)
        ids = {h["doc_id"] for h in hits}
        r1, r2 = a.ask(q["question"], q["role"]), a.ask(q["question"], q["role"])
        t = q["type"]
        if t in ("answerable", "conflict"):
            hit = bool(ids & set(q["expected_docs"])); ret.append(hit)
            text = r1["answer"].lower()
            ok = all(k.lower() in text for k in q["keywords"]) and r1["status"] in ("ok", "conflict")
            acc.append(ok)
            g = any(s["cited"] for s in r1["sources"]); grd.append(g)
            cite = lambda r: {s["doc_id"] for s in r["sources"] if s["cited"]}
            c = r1["status"] == r2["status"] and cite(r1) == cite(r2)
            con.append(c); res = f"retr={hit} acc={ok} grounded={g} consistent={c}"
        elif t == "restricted":
            leak = ids & set(q["forbidden_docs"]) or {s["doc_id"] for s in r1["sources"]} & set(q["forbidden_docs"])
            safe.append(not leak); res = f"no_leak={not leak}"
        else:
            ok = r1["status"] == "insufficient_evidence"; safe.append(ok); res = f"insufficient_handled={ok}"
        rows.append(f"| {q['id']} | {t} | {q['role']} | {res} |")
    errs = [a.ask("   ", "claims_handler")["status"] == "error"]
    try: a.ask("test", "hacker"); errs.append(False)
    except PermissionError: errs.append(True)
    try: a.search("", "claims_handler"); errs.append(False)
    except ValueError: errs.append(True)
    pct = lambda l: f"{sum(l)}/{len(l)}"
    md = [f"# Test Report ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
          "Measured by `python eval/run_eval.py` against live AWS Bedrock.", "",
          "| Metric | Result |", "|---|---|",
          f"| Retrieval relevance (expected doc in top-{TOP_K}) | {pct(ret)} |",
          f"| Answer accuracy (keywords + valid status) | {pct(acc)} |",
          f"| Grounding (answer cites a source) | {pct(grd)} |",
          f"| Response consistency (2 runs, same status + cited docs) | {pct(con)} |",
          f"| Access control / insufficient-evidence safety | {pct(safe)} |",
          f"| Error handling (empty question, bad role, empty search) | {pct(errs)} |", "",
          "| ID | Type | Role | Result |", "|---|---|---|---|", *rows, "",
          "## Known limitations",
          "- Keyword-based accuracy check is a proxy; review answers manually.",
          "- Vector index is a local persisted cosine index sized for a small corpus.",
          "- MIN_SCORE threshold needs tuning per corpus.",
          "- Role comes from the UI selector / MCP env var; production needs a real identity source."]
    (Path(__file__).parent / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
