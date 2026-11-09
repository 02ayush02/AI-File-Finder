import os
import pytest

pytest.importorskip("streamlit")
import streamlit as st
from streamlit.testing.v1 import AppTest


def test_app_search_flow(corpus, tmp_path, monkeypatch):
    root, made = corpus
    monkeypatch.setenv("FILEFINDER_WATCH_DIRS", str(root))
    monkeypatch.setenv("FILEFINDER_STORAGE_DIR", str(tmp_path / "st"))
    for k, v in dict(EMBED_BACKEND="hashing", LLM_BACKEND="heuristic", RERANKER_BACKEND="lexical").items():
        monkeypatch.setenv("FILEFINDER_" + k, v)
    import importlib, config
    importlib.reload(config)              # app.py reads config.settings at import time; pick up the env above
    st.cache_resource.clear()
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    import time
    n_expected = sum(1 for p in root.rglob('*') if p.is_file())
    for _ in range(60):                      # wait for the background initial scan to finish
        at.run()
        files = [int(m.value) for m in at.metric if m.label == "Files"]
        if files and files[0] >= n_expected and not at.sidebar.info:
            break
        time.sleep(0.5)
    assert files and files[0] >= n_expected
    at.chat_input[0].set_value("the quote with part number XJ-4471-B").run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "scan_0091.pdf" in text and "XJ-4471-B" in text
    assert any(s.value.startswith("Found") for s in at.success)
