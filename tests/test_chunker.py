import re
from common import Segment, est_tokens
from indexer.chunker import chunk_segments, format_location


def _long_pages(n_pages=6, sentences=60):
    segs = []
    for p in range(1, n_pages + 1):
        text = " ".join(f"Page{p} sentence number {i} talks about turbine blade fatigue and creep." for i in range(sentences))
        segs.append(Segment(text=text, unit="Page", start=p, end=p))
    return segs


def test_chunk_sizes_and_overlap():
    segs = _long_pages()
    chunks = chunk_segments(segs, chunk_tokens=600, overlap_ratio=0.17)
    assert len(chunks) > 3
    for c in chunks:
        assert c.tokens <= 650, c.tokens          # never (much) above the target
    # consecutive chunks share text (overlap), and nothing is lost
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-120:]
        if tail.split(".")[-2][-40:] in b.text:
            shared += 1
    assert shared >= len(chunks) - 2
    joined = " ".join(c.text for c in chunks)
    for p in range(1, 7):
        for i in (0, 30, 59):
            assert f"Page{p} sentence number {i} " in joined


def test_sentences_not_cut_mid_way():
    chunks = chunk_segments(_long_pages(2, 80), chunk_tokens=500, overlap_ratio=0.15)
    for c in chunks:
        assert c.text.rstrip().endswith("creep."), c.text[-60:]


def test_locations_track_pages():
    chunks = chunk_segments(_long_pages(4, 40), chunk_tokens=500)
    assert chunks[0].location.startswith("Page")
    assert any("Pages" in c.location for c in chunks) or all(c.location.startswith("Page") for c in chunks)


def test_small_segments_merge_and_location_range():
    segs = [Segment(f"slide {i} text about budgets", "Slide", i, i) for i in range(1, 4)]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1 and chunks[0].location == "Slides 1-3"


def test_format_location_variants():
    assert format_location([Segment("x", "Rows", 41, 80, "Sheet 'Q3'")]) == "Sheet 'Q3' rows 41-80"
    assert format_location([Segment("x", "Section", 1, 1, "Intro")]) == "Section 'Intro'"
    assert format_location([Segment("x", "Lines", 1, 40)]) == "Lines 1-40"


def test_giant_unbroken_text_is_hard_split():
    chunks = chunk_segments([Segment("x" * 20000, "Lines", 1, 1)], chunk_tokens=500)
    assert len(chunks) > 5 and all(c.tokens <= 600 for c in chunks)


def test_empty():
    assert chunk_segments([]) == []
