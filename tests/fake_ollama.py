"""A tiny fake Ollama HTTP server so the real Ollama code paths can be tested without models."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from indexer.embedders import HashingEmbedder


class FakeOllama:
    def __init__(self, models=("nomic-embed-text:latest", "qwen2.5:7b"), chat_fn=None):
        self.models = list(models)
        self.chat_fn = chat_fn or (lambda system, user: {})
        self.embed_inputs: list[str] = []
        self.chat_calls: list[tuple[str, str]] = []
        self._emb = HashingEmbedder(128)
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send({"models": [{"name": m} for m in outer.models]})
                else:
                    self._send({}, 404)

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/embed":
                    inp = data["input"]
                    outer.embed_inputs.extend(inp)
                    self._send({"embeddings": [outer._emb._embed(t) for t in inp]})
                elif self.path == "/api/chat":
                    system = data["messages"][0]["content"]; user = data["messages"][1]["content"]
                    outer.chat_calls.append((system, user))
                    self._send({"message": {"content": json.dumps(outer.chat_fn(system, user))}})
                else:
                    self._send({}, 404)

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
