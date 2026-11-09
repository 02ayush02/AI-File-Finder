"""pytest configuration: puts the project root on sys.path and provides shared fixtures."""
import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """A generated messy file collection (see scripts/make_sample_corpus.py)."""
    from make_sample_corpus import make
    root = tmp_path_factory.mktemp("corpus")
    made = make(str(root))
    return root, made


@pytest.fixture()
def make_settings(tmp_path):
    from config import load_settings

    def _make(watch_dirs, **kw):
        base = dict(watch_dirs=[str(d) for d in watch_dirs], storage_dir=str(tmp_path / "storage"),
                    embed_backend="hashing", llm_backend="heuristic", reranker_backend="lexical",
                    watch_debounce_s=0.4)
        base.update(kw)
        return load_settings(**base)
    return _make
