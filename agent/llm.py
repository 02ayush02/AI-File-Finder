"""LLM access for the planner and verifier. Only Ollama is supported as a real backend (local, free).

If no LLM is reachable ("auto" mode), `get_llm` returns None and the planner/verifier use deterministic
rule-based fallbacks - the system keeps working, just with less language understanding.
"""
from __future__ import annotations

from typing import Optional

import ollama_client
from common import get_logger

log = get_logger("llm")


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, url: str, timeout: float = 120.0):
        self.model, self.url, self.timeout = model, url, timeout

    def chat_json(self, system: str, user: str) -> dict:
        return ollama_client.chat_json(self.url, self.model, system, user, timeout=self.timeout)

    def __repr__(self) -> str:
        return f"OllamaLLM({self.model})"


def get_llm(settings) -> Optional[OllamaLLM]:
    backend = settings.llm_backend.lower()
    if backend == "heuristic":
        return None
    if backend not in ("auto", "ollama"):
        raise ValueError(f"Unknown llm_backend: {settings.llm_backend}")
    if ollama_client.has_model(settings.ollama_url, settings.llm_model):
        log.info("LLM agent: Ollama model '%s'", settings.llm_model)
        return OllamaLLM(settings.llm_model, settings.ollama_url, settings.llm_timeout_s)
    if backend == "ollama":
        raise RuntimeError(f"Ollama model '{settings.llm_model}' not available at {settings.ollama_url}. "
                           f"Run: ollama pull {settings.llm_model}")
    log.warning("No Ollama LLM '%s' reachable -> using the rule-based query planner / verifier "
                "(install Ollama and `ollama pull %s` for full agentic behaviour).",
                settings.llm_model, settings.llm_model)
    return None
