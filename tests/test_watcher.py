import os, shutil
from agent.pipeline import FileFinder
from indexer.file_watcher import FileWatcher


def test_watcher_lifecycle(make_settings, tmp_path):
    watch = tmp_path / "w"; watch.mkdir()
    f = FileFinder(make_settings([watch]))
    w = FileWatcher(f.store, f.s); w.start(); assert w.wait_idle()
    names = lambda: sorted(r["name"] for r in f.store.list_files())
    try:
        (watch / "n.txt").write_text("hydrogen electrolyzer stack and deionized water loop"); w.wait_idle()
        assert names() == ["n.txt"]
        (watch / "n.txt").write_text("sourdough hydration and proofing"); w.wait_idle()
        assert f.search("hydrogen electrolyzer", verify=False).matches == [] or \
               f.search("hydrogen electrolyzer", verify=False).matches[0].score < 0.2
        os.makedirs(watch / "sub"); shutil.move(watch / "n.txt", watch / "sub" / "m.txt"); w.wait_idle()
        assert names() == ["m.txt"]
        (watch / "sub" / "m.txt").unlink(); w.wait_idle()
        assert names() == []
        (watch / "~$lock.docx").write_text("temp"); (watch / "x.tmp").write_text("temp"); w.wait_idle()
        assert names() == []                                # temp/lock files ignored
        os.makedirs(watch / "proj")
        for i in range(3):
            (watch / "proj" / f"{i}.txt").write_text(f"note {i} gearbox")
        w.wait_idle(); assert len(names()) == 3
        shutil.rmtree(watch / "proj"); w.wait_idle(); assert names() == []
        assert f.store.consistency_check()["orphan_vectors"] == 0
    finally:
        w.stop(); f.close()


def test_initial_scan_removes_files_deleted_while_offline(make_settings, tmp_path):
    watch = tmp_path / "w"; watch.mkdir()
    (watch / "a.txt").write_text("alpha words"); (watch / "b.txt").write_text("beta words")
    f = FileFinder(make_settings([watch])); f.index_all()
    (watch / "a.txt").unlink()
    w = FileWatcher(f.store, f.s); w.start(); w.wait_idle()
    assert [r["name"] for r in f.store.list_files()] == ["b.txt"]
    w.stop(); f.close()
