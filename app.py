"""
Streamlit UI:   streamlit run app.py

* Starts the file watcher inside this process (so the vector DB has a single writer) and runs the initial scan in
  the background - you can search while it is still indexing.
* Describe the file in plain language; get the exact path, page/section, and a quote proving the match.
"""
from __future__ import annotations

import datetime as dt
import os

import streamlit as st

from config import settings
from opener import open_file, reveal_in_folder

st.set_page_config(page_title="AI File Finder", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading models and opening the index …")
def get_system():
    from agent.pipeline import FileFinder
    from indexer.file_watcher import FileWatcher

    finder = FileFinder(settings)
    watcher = FileWatcher(finder.store, settings)
    watcher.start()
    return finder, watcher


finder, watcher = get_system()


# ------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Index")
    status = finder.status()
    c1, c2 = st.columns(2)
    c1.metric("Files", status["files"])
    c2.metric("Chunks", status["chunks"])
    if watcher.scanning:
        st.info("Initial scan running … you can already search what is indexed so far.")
    elif watcher.scan_stats is not None:
        st.caption("Last scan: " + watcher.scan_stats.summary())
    st.caption(f"Embeddings: `{status['embedder']}`")
    st.caption(f"Reranker: `{status['reranker']}`  \nAgent LLM: `{status['llm']}`")
    if not status["semantic_embeddings"]:
        st.warning("Offline hashing embeddings are active (lexical, not semantic). "
                   "Install Ollama and `ollama pull nomic-embed-text` for true semantic search.")
    with st.expander("Watched folders"):
        for d in status["watch_dirs"]:
            st.write(("✅ " if os.path.isdir(d) else "⚠️ missing: ") + d)
    if st.button("Re-scan now", disabled=watcher.scanning):
        with st.spinner("Scanning …"):
            stats = finder.index_all()
        st.success(stats.summary())
        st.rerun()

    st.header("Search options")
    top_k = st.slider("Files to show", 1, 10, settings.final_files)
    verify = st.toggle("Verify with agent (and self-correct)", value=True)
    retries = st.slider("Max self-correction rounds", 0, 3, settings.max_retries)
    show_trace = st.toggle("Show how the search was done", value=False)

    with st.expander("Recent index activity"):
        evs = list(watcher.events)[:15]
        if not evs:
            st.caption("No file changes seen yet.")
        for e in evs:
            st.caption(f"{dt.datetime.fromtimestamp(e.ts):%H:%M:%S}  **{e.action}**  {os.path.basename(e.path)}")


# ------------------------------------------------------------------------------------------ rendering
def conf_badge(m) -> str:
    c = m.confidence if m.confidence is not None else m.score
    color = "green" if c >= 0.7 else "orange" if c >= 0.45 else "red"
    return f":{color}[{c:.0%} confidence]"


def render_response(resp, key_prefix: str, show_trace_flag: bool) -> None:
    (st.success if resp.confident else st.warning)(resp.message)
    for i, m in enumerate(resp.matches, start=1):
        with st.container(border=True):
            top, right = st.columns([5, 2])
            with top:
                st.markdown(f"**{i}. {m.file_name}** &nbsp; {conf_badge(m)} &nbsp; `{m.extension}` · modified {m.modified}")
                st.code(m.file_path, language=None)
            with right:
                b1, b2 = st.columns(2)
                if b1.button("Open", key=f"{key_prefix}-open-{i}", use_container_width=True):
                    try:
                        open_file(m.file_path)
                    except Exception as e:
                        st.error(f"Could not open: {e}")
                if b2.button("Folder", key=f"{key_prefix}-dir-{i}", use_container_width=True):
                    try:
                        reveal_in_folder(m.file_path)
                    except Exception as e:
                        st.error(f"Could not reveal: {e}")
                st.markdown(f"[file link]({m.open_uri})")
            if m.reason:
                st.markdown(f"**Why:** {m.reason}")
            if m.evidence:
                loc = f" — *{m.evidence_location}*" if m.evidence_location else ""
                st.markdown(f"**Evidence**{loc}")
                st.markdown("> " + m.evidence.replace("\n", "\n> "))
            if len(m.hits) > 1:
                with st.expander(f"{len(m.hits) - 1} more matching passage(s)"):
                    for h in m.hits[1:]:
                        st.caption(f"{h.location}  (score {h.score:.2f})")
                        st.markdown("> " + h.snippet)
    if show_trace_flag and resp.attempts:
        with st.expander("How I searched", expanded=False):
            for n, a in enumerate(resp.attempts, start=1):
                p = a.plan
                st.markdown(f"**Attempt {n}** — planner `{p.planner}`" + (f" — {p.notes}" if p.notes else ""))
                st.markdown(f"- semantic query: `{p.semantic_query}`\n- exact keywords: `{p.exact_keywords}`\n"
                            f"- filters: {p.filters.describe()}\n- {a.note}\n"
                            f"- verifier `{a.verifier}` → confident: **{a.confident}** (top {a.top_confidence:.2f})")
    st.caption(f"{resp.elapsed_s:.2f}s")


# ------------------------------------------------------------------------------------------ main
st.title("🔎 AI File Finder")
st.caption("Describe the file you are looking for — what it contains, roughly when you worked on it, what type it is.")

if "history" not in st.session_state:
    st.session_state.history = []

for n, (q, resp) in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        render_response(resp, f"h{n}", show_trace)

query = st.chat_input("e.g. “the doc from last month with the Q3 pump efficiency calculations”")
if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Planning → searching → reranking → verifying …"):
            resp = finder.search(query, top_k=top_k, verify=verify, max_retries=retries)
        render_response(resp, f"h{len(st.session_state.history)}", show_trace)
    st.session_state.history.append((query, resp))
