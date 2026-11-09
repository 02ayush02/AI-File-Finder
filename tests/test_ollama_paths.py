import os, re
from datetime import datetime
from agent.pipeline import FileFinder
from indexer.embedders import OllamaEmbedder, get_embedder
from tests.fake_ollama import FakeOllama


def test_embedder_auto_picks_ollama_with_prefixes(make_settings, tmp_path):
    with FakeOllama() as fo:
        s = make_settings([tmp_path], embed_backend="auto", ollama_url=fo.url, embed_model="nomic-embed-text")
        emb = get_embedder(s)
        assert isinstance(emb, OllamaEmbedder) and emb.dim == 128
        emb.embed_documents(["hello"]); emb.embed_query("question")
        assert "search_document: hello" in fo.embed_inputs and "search_query: question" in fo.embed_inputs


def test_embedder_falls_back_when_ollama_down(make_settings, tmp_path):
    s = make_settings([tmp_path], embed_backend="auto", ollama_url="http://127.0.0.1:9")
    assert get_embedder(s).backend == "hashing"
    s = make_settings([tmp_path], embed_backend="ollama", ollama_url="http://127.0.0.1:9")
    try:
        get_embedder(s); assert False
    except RuntimeError as e:
        assert "ollama pull" in str(e)


def test_full_agentic_flow_with_llm(corpus, make_settings):
    root, made = corpus
    quote = "Item 1: Deep groove ball bearing, part number XJ-4471-B"

    def chat(system, user):
        if "convert a user's description" in system:
            return {"semantic_query": "supplier quotation bearing part number", "exact_keywords": ["XJ-4471-B"],
                    "modified_after": "2099-01-01", "modified_before": None, "file_types": ["pdf"], "path_contains": None}
        # verifier: pick whichever candidate mentions the part number; fabricate a quote for it
        blocks = re.split(r"\n\n(?=\[\d+\])", user.split("CANDIDATES:\n")[1])
        res, ok = [], False
        for b in blocks:
            i = int(re.match(r"\[(\d+)\]", b).group(1))
            if "XJ-4471-B" in b:
                res.append({"id": i, "matches": True, "confidence": 0.93, "reason": "has the part number",
                            "evidence_quote": quote}); ok = True
            else:
                res.append({"id": i, "matches": False, "confidence": 0.1, "reason": "no", "evidence_quote": "made up"})
        return {"results": res, "retry": None}

    with FakeOllama(chat_fn=chat) as fo:
        s = make_settings([root], embed_backend="auto", llm_backend="auto", ollama_url=fo.url,
                          embed_model="nomic-embed-text", llm_model="qwen2.5:7b")
        f = FileFinder(s)
        assert f.llm is not None and f.embedder.backend == "ollama"
        f.index_all()
        r = f.search("the PDF quote with part XJ-4471-B")
        # LLM planner set an impossible date filter (2099) -> attempt 1 finds nothing -> self-correction drops it
        assert len(r.attempts) >= 2 and "filters" in r.attempts[1].plan.notes.lower() or r.attempts[1].plan.planner == "relaxed"
        assert r.confident and os.path.basename(r.matches[0].file_path) == "scan_0091.pdf"
        assert r.matches[0].evidence == quote            # genuine quote kept
        assert r.attempts[0].plan.planner.startswith("llm")
        assert all("made up" not in (m.evidence or "") for m in r.matches)  # fabricated quote replaced by real text


def test_llm_rewrite_retry(corpus, make_settings):
    root, _ = corpus
    state = {"n": 0}

    def chat(system, user):
        if "convert a user's description" in system:
            return {"semantic_query": "zzzz qqqq", "exact_keywords": [], "modified_after": None,
                    "modified_before": None, "file_types": [], "path_contains": None}
        state["n"] += 1
        if state["n"] == 1:
            return {"results": [{"id": 1, "matches": False, "confidence": 0.05, "reason": "unrelated"}],
                    "retry": {"semantic_query": "sourdough starter feeding", "exact_keywords": [], "drop_filters": True}}
        return {"results": [{"id": 1, "matches": True, "confidence": 0.9, "reason": "recipe", "evidence_quote": ""}],
                "retry": None}

    with FakeOllama(chat_fn=chat) as fo:
        s = make_settings([root], embed_backend="auto", llm_backend="auto", ollama_url=fo.url)
        f = FileFinder(s); f.index_all()
        r = f.search("that bread thing")
        assert r.confident and os.path.basename(r.matches[0].file_path) == "untitled.md"
        assert r.attempts[1].plan.planner.startswith("rewrite")


def test_llm_garbage_falls_back_to_heuristics(corpus, make_settings):
    root, _ = corpus
    with FakeOllama(chat_fn=lambda s, u: {"unexpected": 1}) as fo:
        s = make_settings([root], embed_backend="auto", llm_backend="auto", ollama_url=fo.url)
        f = FileFinder(s); f.index_all()
        r = f.search("sourdough starter feeding schedule")
        assert os.path.basename(r.matches[0].file_path) == "untitled.md" and r.confident
