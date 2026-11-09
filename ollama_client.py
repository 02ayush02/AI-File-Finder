"""Tiny Ollama HTTP client (embeddings + JSON chat). Uses only `requests`."""
from __future__ import annotations

import json
import re
from typing import Optional

import requests

from common import get_logger

log = get_logger("ollama")


class OllamaError(RuntimeError):
    pass


def list_models(base_url: str, timeout: float = 3.0) -> Optional[list[str]]:
    """Names of locally pulled models, or None if the server is unreachable."""
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return None


def has_model(base_url: str, model: str, timeout: float = 3.0) -> bool:
    models = list_models(base_url, timeout)
    if models is None:
        return False
    want = model if ":" in model else model + ":latest"
    return any(m == model or m == want or m.split(":")[0] == model for m in models)


def embed(base_url: str, model: str, texts: list[str], timeout: float = 300.0) -> list[list[float]]:
    url = base_url.rstrip("/")
    try:
        r = requests.post(url + "/api/embed", json={"model": model, "input": texts, "truncate": True},
                          timeout=timeout)
        if r.status_code == 404:  # very old Ollama: single-prompt endpoint
            out = []
            for t in texts:
                rr = requests.post(url + "/api/embeddings", json={"model": model, "prompt": t}, timeout=timeout)
                rr.raise_for_status()
                out.append(rr.json()["embedding"])
            return out
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings")
        if not embs or len(embs) != len(texts):
            raise OllamaError(f"Unexpected /api/embed response: {str(data)[:200]}")
        return embs
    except requests.RequestException as e:
        raise OllamaError(f"Ollama embedding request failed: {e}") from e


def chat_json(base_url: str, model: str, system: str, user: str, timeout: float = 120.0,
              temperature: float = 0.0) -> dict:
    """Ask the model for a JSON object. Robust to models that wrap JSON in prose / code fences."""
    try:
        r = requests.post(
            base_url.rstrip("/") + "/api/chat",
            json={
                "model": model, "stream": False, "format": "json",
                "options": {"temperature": temperature, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
    except requests.RequestException as e:
        raise OllamaError(f"Ollama chat request failed: {e}") from e
    return extract_json(content)


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            v = json.loads(text[start:end + 1])
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    raise OllamaError(f"Model did not return valid JSON: {text[:200]!r}")
