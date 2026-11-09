import os
import pytest
from agent.pipeline import FileFinder


@pytest.fixture()
def finder(corpus, make_settings):
    root, made = corpus
    f = FileFinder(make_settings([root]))
    f.index_all()
    yield f, made
    f.close()


CASES = [
    ("Find the doc with the Q3 pump efficiency calculations", "pump_xlsx"),
    ("Find that document from last month where I calculated the stress tensor for the gear shaft", "shaft_docx"),
    ("the quote with part number XJ-4471-B", "quote_pdf"),
    ("scanned insurance policy with policy number HX-99231", "scanned_pdf"),
    ("receipt from the hardware store", "receipt_img"),
    ("python function converting pressure to pump head", "py"),
    ("sourdough starter feeding", "sourdough"),
]


@pytest.mark.parametrize("query,key", CASES)
def test_top1(finder, query, key):
    f, made = finder
    r = f.search(query)
    assert r.matches[0].file_path == str(made[key]), [m.file_name for m in r.matches]
    assert r.confident and r.matches[0].evidence


def test_date_filter_excludes_old_file(finder):
    f, made = finder
    r = f.search("gear shaft notes from last month", max_retries=0)
    names = [m.file_name for m in r.matches]
    assert "gear_notes.txt" not in names       # 240 days old
    r2 = f.search("gear shaft notes", max_retries=0)
    assert "gear_notes.txt" in [m.file_name for m in r2.matches]


def test_file_type_filter(finder):
    f, made = finder
    r = f.search("pdf with bearing quote", max_retries=0)
    assert all(m.extension == ".pdf" for m in r.matches)


def test_unanswerable_is_not_confident(finder):
    f, _ = finder
    r = f.search("my 2019 tax return and mortgage statement")
    assert not r.confident and "No file clearly matches" in r.message
    assert len(r.attempts) >= 2             # tried to self-correct


def test_results_grouped_by_file_and_locations(finder):
    f, made = finder
    r = f.search("pump")
    paths = [m.file_path for m in r.matches]
    assert len(paths) == len(set(paths))
    assert all(m.hits and m.hits[0].location for m in r.matches)


def test_deleted_file_never_returned(finder, tmp_path):
    import shutil
    f, made = finder
    backup = tmp_path / "backup.md"
    shutil.copy2(made["sourdough"], backup)           # the corpus is shared by the whole test session: restore it
    try:
        os.remove(made["sourdough"])
        r = f.search("sourdough starter feeding")
        assert str(made["sourdough"]) not in [m.file_path for m in r.matches]
        assert f.store.get_file(str(made["sourdough"])) is None      # lazily purged
    finally:
        shutil.copy2(backup, made["sourdough"])


def test_empty_index_and_empty_query(make_settings, tmp_path):
    d = tmp_path / "empty"; d.mkdir()
    f = FileFinder(make_settings([d]))
    assert "empty" in f.search("anything").message.lower()
    assert "describe" in f.search("  ").message.lower()


def test_open_uri(finder):
    f, made = finder
    r = f.search("sourdough starter feeding")
    assert r.matches[0].open_uri.startswith("file://")
