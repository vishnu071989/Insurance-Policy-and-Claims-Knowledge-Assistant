import streamlit as st
from access import ROLES
from rag import Assistant

st.set_page_config(page_title="Insurance Knowledge Assistant", page_icon="📄")
st.title("Insurance Policy & Claims Knowledge Assistant")

@st.cache_resource
def assistant():
    return Assistant()

role = st.sidebar.selectbox("Signed-in role (simulated sign-in)", sorted(ROLES))
if st.session_state.get("role") != role:
    st.session_state.update(role=role, msgs=[])
st.sidebar.caption("Answers come only from approved documents your role may access. "
                   "This tool does not make coverage or claim decisions.")

starters = {"Coverage": "Is a burst pipe covered under the homeowner policy?",
            "Exclusions": "What does the homeowner policy exclude?",
            "Documentation": "Which documents are required to file a claim?",
            "Claims process": "What is the claims processing procedure?"}
cols = st.columns(len(starters))
for col, (label, q) in zip(cols, starters.items()):
    if col.button(label, use_container_width=True):
        st.session_state["pending"] = q

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**[{s['ref']}] {s['doc_id']} - {s['title']} - {s['section']}**"
                                f" (score {s['score']}{', cited' if s['cited'] else ''})")
                    st.caption(s["text"])

q = st.chat_input("Ask about coverage, exclusions, documentation or claims process") or st.session_state.pop("pending", None)
if q:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs]
    st.session_state.msgs.append({"role": "user", "content": q})
    try:
        r = assistant().ask(q, role, history)
    except Exception as e:
        r = {"status": "error", "answer": f"Error: {e}", "sources": []}
    badge = {"ok": "", "conflict": "⚠️ **Conflicting sources**\n\n",
             "insufficient_evidence": "ℹ️ **Insufficient evidence**\n\n", "error": "❌ "}[r["status"]]
    st.session_state.msgs.append({"role": "assistant", "content": badge + r["answer"], "sources": r["sources"]})
    st.rerun()
