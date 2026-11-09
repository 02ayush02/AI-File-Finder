import sys, types
from agent.reranker import get_reranker, LexicalReranker, CrossEncoderReranker


def test_lexical_reranker_prefers_exact_matches():
    r = LexicalReranker()
    a, b = r.score("pump efficiency Q3", ["Q3 pump efficiency calculation table", "holiday photos of cats"], ["Q3"])
    assert a > 0.6 and b < 0.1


def test_cross_encoder_logits_are_squashed(monkeypatch):
    class FakeCE:
        def __init__(self, name, max_length=512): self.name, self.max_length = name, max_length
        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            return [4.0 if "pump" in p[1] else -6.0 for p in pairs]
    mod = types.ModuleType("sentence_transformers"); mod.CrossEncoder = FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    from config import load_settings
    rr = get_reranker(load_settings(reranker_backend="cross_encoder"))
    assert isinstance(rr, CrossEncoderReranker) and "bge-reranker-v2-m3" in rr.name
    hi, lo = rr.score("pump", ["a pump", "a cat"])
    assert 0.98 < hi <= 1.0 and 0.0 <= lo < 0.01


def test_auto_falls_back_when_model_cannot_load(monkeypatch):
    class Boom:
        def __init__(self, *a, **k): raise OSError("no network")
    mod = types.ModuleType("sentence_transformers"); mod.CrossEncoder = Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    from config import load_settings
    assert isinstance(get_reranker(load_settings(reranker_backend="auto")), LexicalReranker)
