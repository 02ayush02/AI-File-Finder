import os, shutil, time
from common import Filters
from indexer.embedders import HashingEmbedder, get_embedder
from indexer.embed_store import IndexStore, build_fts_query


def mk(make_settings, tmp_path, **kw):
    watch = tmp_path / "w"; watch.mkdir(exist_ok=True)
    s = make_settings([watch], **kw)
    return watch, s, IndexStore(s, get_embedder(s))


def test_dedup_touch_modify_delete(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    f = watch / "a.txt"; f.write_text("alpha beta gamma pumps and valves")
    assert st.index_file(str(f)).status == "indexed"
    assert st.index_file(str(f)).status == "unchanged"
    t0 = st.get_file(str(f))["indexed_at"]
    os.utime(f, (time.time() + 100, time.time() + 100))
    assert st.index_file(str(f)).status == "metadata_updated"      # same hash -> no re-embedding
    assert st.get_file(str(f))["indexed_at"] == t0
    f.write_text("completely new text about compressors")
    assert st.index_file(str(f)).status == "indexed"
    assert st.sparse_search(build_fts_query(terms=["alpha"]), 5) == []
    assert st.sparse_search(build_fts_query(terms=["compressors"]), 5)
    assert st.remove_file(str(f)) and st.stats()["chunks"] == 0
    assert st.consistency_check() == {"orphan_vectors": 0, "chunks_without_vector": 0, "repaired": False}


def test_filters_dense_and_sparse(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    old, new = watch / "old.txt", watch / "new.md"
    old.write_text("hydraulic pump efficiency"); new.write_text("hydraulic pump efficiency")
    os.utime(old, (time.time() - 90 * 86400,) * 2)
    st.index_file(str(old)); st.index_file(str(new))
    emb = st.embedder.embed_query("hydraulic pump efficiency")
    cut = Filters(modified_after=time.time() - 10 * 86400)
    assert len(st.dense_search(emb, 10)) == 2 and len(st.dense_search(emb, 10, cut)) == 1
    assert len(st.sparse_search(build_fts_query(terms=["pump"]), 10, cut)) == 1
    assert len(st.sparse_search(build_fts_query(terms=["pump"]), 10, Filters(file_types=[".txt"]))) == 1
    assert len(st.dense_search(emb, 10, Filters(path_contains="new.m"))) == 1
    both = Filters(modified_after=time.time() - 10 * 86400, file_types=[".md"])
    assert len(st.dense_search(emb, 10, both)) == 1


def test_exact_part_number_phrase(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    (watch / "p.txt").write_text("order bearing XJ-4471-B quantity forty")
    (watch / "q.txt").write_text("order bearing XJ-9999-B quantity forty")
    st.index_file(str(watch / "p.txt")); st.index_file(str(watch / "q.txt"))
    hits = st.sparse_search(build_fts_query(phrases=["XJ-4471-B"]), 5)
    assert len(hits) == 1


def test_missing_and_moved(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    a = watch / "a.txt"; a.write_text("some searchable words here")
    st.index_file(str(a))
    b = watch / "b.txt"; shutil.move(a, b)
    st.move_file(str(a), str(b))
    assert [r["name"] for r in st.list_files()] == ["b.txt"]
    b.unlink()
    assert st.remove_missing([str(watch)]) == 1 and st.stats()["files"] == 0


def test_unmounted_root_keeps_entries(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    a = watch / "a.txt"; a.write_text("keep me around")
    st.index_file(str(a))
    shutil.rmtree(watch)
    assert st.remove_missing([str(watch)]) == 0       # root missing => assume unmounted, do not wipe


def test_embedder_change_rebuilds(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    (watch / "a.txt").write_text("some words"); st.index_file(str(watch / "a.txt")); st.close()
    s2 = make_settings([watch], hashing_dim=256)
    st2 = IndexStore(s2, get_embedder(s2))
    assert st2.stats()["files"] == 0                # old derived index discarded
    assert st2.index_file(str(watch / "a.txt")).status == "indexed"


def test_filtered_and_empty_files(make_settings, tmp_path):
    watch, s, st = mk(make_settings, tmp_path)
    (watch / "z.bin").write_bytes(b"123"); (watch / "e.txt").write_text("")
    (watch / ".hidden").mkdir(); (watch / ".hidden" / "x.txt").write_text("secret")
    assert st.index_file(str(watch / "z.bin")).status == "skipped"
    assert st.index_file(str(watch / "e.txt")).status == "skipped"
    assert st.scan([str(watch)]).seen == 0


def test_file_without_text_still_findable_by_name(make_settings, tmp_path):
    from PIL import Image
    watch, s, st = mk(make_settings, tmp_path, ocr_backend="none")
    Image.new("RGB", (200, 200), "red").save(watch / "holiday_photo_beach.png")
    assert st.index_file(str(watch / "holiday_photo_beach.png")).status == "indexed"
    assert st.sparse_search(build_fts_query(terms=["beach"]), 5)
