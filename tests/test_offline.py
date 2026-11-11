"""Offline logic tests with stub embedder/LLM. These do NOT validate Bedrock."""
import sys, tempfile, unittest, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
import ingest
from rag import Assistant

def stub_embed(t):
    v = np.zeros(256, dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", t.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return v

def stub_llm(system, user, max_tokens=0):
    return "Per the passages [S1]."

class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ingest.run(index_dir=tempfile.mkdtemp(), embed_fn=stub_embed)
        cls.a = Assistant(store=cls.store, embed_fn=stub_embed, llm_fn=stub_llm)
    def test_chunks(self):
        self.assertGreater(len(self.store.chunks), 10)
    def test_acl_restricted_hidden(self):
        ids = {h["doc_id"] for h in self.a.search("SIU referral red flags", "customer_support", 50)}
        self.assertNotIn("CLM-003", ids); self.assertNotIn("UW-001", ids)
        ids = {h["doc_id"] for h in self.a.search("SIU referral red flags", "claims_handler", 50)}
        self.assertIn("CLM-003", ids)
    def test_bad_role(self):
        with self.assertRaises(PermissionError): self.a.ask("x", "admin")
    def test_empty(self):
        self.assertEqual(self.a.ask(" ", "claims_handler")["status"], "error")
    def test_ungrounded_discarded(self):
        a = Assistant(store=self.store, embed_fn=stub_embed, llm_fn=lambda s, u, m=0: "Yes it is covered.")
        self.assertEqual(a.ask("burst pipe covered plumbing water", "claims_handler")["status"], "insufficient_evidence")
    def test_grounded(self):
        r = self.a.ask("burst pipe covered plumbing water", "claims_handler")
        self.assertEqual(r["status"], "ok"); self.assertTrue(r["sources"][0]["cited"])
    def test_no_hits_skips_llm(self):
        a = Assistant(store=self.store, embed_fn=stub_embed, llm_fn=lambda *x: 1/0)
        self.assertEqual(a.ask("zzzz qqqq", "claims_handler")["status"], "insufficient_evidence")

if __name__ == "__main__":
    unittest.main()
