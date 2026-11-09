#!/usr/bin/env python3
"""
Zero-setup demo: generates a sample file collection, indexes it into a throw-away location using the offline
fallback backends (no Ollama / no model downloads needed), and runs example searches.

    python scripts/demo.py
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_corpus import make          # noqa: E402
from config import load_settings             # noqa: E402
from agent.pipeline import FileFinder        # noqa: E402

QUERIES = [
    "Find the doc with the Q3 pump efficiency calculations",
    "Find that document from last month where I calculated the stress tensor for the gear shaft",
    "the quote that lists part number XJ-4471-B",
    "scanned insurance policy, policy number HX-99231",
    "photo of a receipt from the hardware store",
    "my 2019 tax return",
]

if __name__ == "__main__":
    corpus = tempfile.mkdtemp(prefix="ff_corpus_")
    make(corpus)
    s = load_settings(watch_dirs=[corpus], storage_dir=tempfile.mkdtemp(prefix="ff_store_"),
                      embed_backend="hashing", llm_backend="heuristic", reranker_backend="lexical")
    finder = FileFinder(s)
    print("Indexing sample files ...", finder.index_all().summary())
    for q in QUERIES:
        r = finder.search(q, top_k=2)
        print(f"\n> {q}\n  {r.message}")
        for m in r.matches[:1]:
            print(f"  -> {os.path.relpath(m.file_path, corpus)}  [{m.evidence_location}]  confidence {m.confidence:.2f}")
            print(f"     \"{m.evidence[:140]}\"")
    finder.close()
