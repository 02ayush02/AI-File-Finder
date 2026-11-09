"""
Continuous background indexing with `watchdog`.

    created / modified  -> (debounced, waits until the file stops changing) -> re-index if content hash changed
    deleted             -> purge its chunks from SQLite + the vector DB (files AND whole folders)
    moved / renamed     -> purge the old path, index the new one (dirs: re-scan the new location)

Run standalone:   python -m indexer.file_watcher
Or embedded:      FileWatcher(store, settings).start()      (the Streamlit app does this)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from common import get_logger
from indexer.embed_store import IndexResult, IndexStore
from indexer.parser import iter_files, should_index

log = get_logger("watcher")


@dataclass
class _Pending:
    kind: str                 # "upsert" | "delete"
    due: float
    attempts: int = 0
    is_dir: bool = False


@dataclass
class WatchEvent:
    ts: float
    action: str               # indexed | removed | unchanged | failed | skipped
    path: str
    detail: str = ""


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "FileWatcher"):
        self.w = watcher

    def on_created(self, e: FileSystemEvent):
        self.w.queue_upsert(e.src_path, is_dir=e.is_directory)

    def on_modified(self, e: FileSystemEvent):
        if not e.is_directory:
            self.w.queue_upsert(e.src_path)

    def on_deleted(self, e: FileSystemEvent):
        self.w.queue_delete(e.src_path, is_dir=e.is_directory)

    def on_moved(self, e: FileSystemEvent):
        self.w.queue_delete(e.src_path, is_dir=e.is_directory)
        self.w.queue_upsert(e.dest_path, is_dir=e.is_directory)


class FileWatcher:
    def __init__(self, store: IndexStore, settings, on_event: Optional[Callable[[WatchEvent], None]] = None):
        self.store = store
        self.s = settings
        self.on_event = on_event
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observer = None
        self._worker: Optional[threading.Thread] = None
        self._scan_thread: Optional[threading.Thread] = None
        self.events: deque[WatchEvent] = deque(maxlen=200)
        self.scan_stats = None
        self.scanning = False
        self._busy = 0            # events popped from the queue but not yet fully processed

    # ------------------------------------------------------------------ queueing (called from watchdog threads)
    def _fs(self, path: str) -> str:
        return os.fspath(path) if isinstance(path, (str, os.PathLike)) else path.decode()

    def queue_upsert(self, path: str, is_dir: bool = False) -> None:
        path = self._fs(path)
        if not is_dir and not should_index(path, self.s, self.s.watch_dirs):
            return
        with self._lock:
            self._pending[path] = _Pending("upsert", time.time() + self.s.watch_debounce_s, is_dir=is_dir)

    def queue_delete(self, path: str, is_dir: bool = False) -> None:
        path = self._fs(path)
        with self._lock:
            self._pending[path] = _Pending("delete", time.time() + 0.2, is_dir=is_dir)

    # ------------------------------------------------------------------ worker
    def _emit(self, action: str, path: str, detail: str = "") -> None:
        ev = WatchEvent(time.time(), action, path, detail)
        self.events.appendleft(ev)
        log.info("%-9s %s %s", action, path, detail)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    def _stable(self, path: str, tries: int = 10, interval: float = 0.5) -> bool:
        """True once the file size/mtime stop changing (file finished copying/saving)."""
        last = None
        for _ in range(tries):
            try:
                st = os.stat(path)
            except OSError:
                return False
            if time.time() - st.st_mtime > 2.0:      # not touched recently: certainly finished
                return True
            cur = (st.st_size, st.st_mtime)
            if cur == last:
                return True
            last = cur
            time.sleep(interval)
        return True

    def _handle(self, path: str, p: _Pending) -> None:
        if p.kind == "delete":
            n = self.store.remove_folder(path)
            removed = self.store.remove_file(path)
            if n or removed:
                self._emit("removed", path, f"{n} file(s) under folder" if n else "")
            return
        if p.is_dir:
            if os.path.isdir(path):
                for fp in iter_files([path], self.s):
                    self._index(fp)
            return
        if not os.path.exists(path):
            return                                  # temp file already gone
        if not self._stable(path):
            return
        self._index(path)

    def _index(self, path: str) -> None:
        res: IndexResult = self.store.index_file(path)
        if res.status == "indexed":
            self._emit("indexed", path, f"{res.n_chunks} chunks" + (f", {res.ocr_pages} OCR page(s)" if res.ocr_pages else ""))
        elif res.status == "metadata_updated":
            self._emit("unchanged", path, "content identical (dates refreshed)")
        elif res.status == "failed":
            raise RuntimeError(res.message)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                due = [(p, v) for p, v in self._pending.items() if v.due <= now]
                for p, _ in due:
                    del self._pending[p]
                self._busy += len(due)
            for path, pend in due:
                try:
                    self._handle(path, pend)
                except Exception as e:
                    pend.attempts += 1
                    if pend.attempts < 3:           # e.g. file still locked by the writer: retry with backoff
                        pend.due = time.time() + 3 * pend.attempts
                        with self._lock:
                            self._pending.setdefault(path, pend)
                    else:
                        self._emit("failed", path, f"{type(e).__name__}: {e}")
                finally:
                    with self._lock:
                        self._busy -= 1
            self._stop.wait(0.3)

    # ------------------------------------------------------------------ lifecycle
    def start(self, initial_scan: Optional[bool] = None) -> None:
        roots = [r for r in self.s.watch_dirs if os.path.isdir(r)]
        if not roots:
            log.warning("No existing directories in WATCH_DIRS (%s); nothing to watch.", self.s.watch_dirs)
        obs_cls = PollingObserver if self.s.watch_polling else Observer
        self._observer = obs_cls()
        handler = _Handler(self)
        for r in roots:
            self._observer.schedule(handler, r, recursive=True)
            log.info("Watching %s", r)
        self._observer.start()
        self._worker = threading.Thread(target=self._run, name="filefinder-watch-worker", daemon=True)
        self._worker.start()
        if self.s.initial_scan_on_start if initial_scan is None else initial_scan:
            self._scan_thread = threading.Thread(target=self._initial_scan, name="filefinder-initial-scan", daemon=True)
            self._scan_thread.start()

    def _initial_scan(self) -> None:
        self.scanning = True
        try:
            log.info("Initial scan of %d folder(s) ...", len(self.s.watch_dirs))
            self.scan_stats = self.store.scan(self.s.watch_dirs)
            log.info("Initial scan done: %s", self.scan_stats.summary())
        except Exception as e:  # pragma: no cover
            log.error("Initial scan failed: %s", e)
        finally:
            self.scanning = False

    def wait_idle(self, timeout: float = 15.0) -> bool:
        """Block until all pending events have been processed (used by tests)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                empty = not self._pending and self._busy == 0
            if empty and not self.scanning:
                time.sleep(0.4)
                with self._lock:
                    if not self._pending and self._busy == 0:
                        return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._worker:
            self._worker.join(timeout=5)


def main() -> None:  # python -m indexer.file_watcher
    from config import settings
    from indexer.embedders import get_embedder

    embedder = get_embedder(settings)
    store = IndexStore(settings, embedder)
    w = FileWatcher(store, settings)
    w.start()
    log.info("Watcher running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        w.stop()
        store.close()


if __name__ == "__main__":
    main()
