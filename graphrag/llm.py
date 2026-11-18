"""Thin LLM wrapper. Two zero-cost backends behind one interface:

* Groq free-tier API (fast; open-weight models such as openai/gpt-oss-120b)
* Ollama running locally (true Llama 3 on your own machine, no key, no limits)
"""
from __future__ import annotations

import json
import re
import time

import requests

from .config import settings


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or settings.llm_provider).lower()
        self.model: str | None = None
        if self.provider == "groq":
            if not settings.groq_api_key or settings.groq_api_key.startswith("gsk_your"):
                raise LLMError("GROQ_API_KEY missing. Add it to .env (free at console.groq.com/keys).")
            from groq import Groq
            self.client = Groq(api_key=settings.groq_api_key)
            self.model = self._pick_groq_model()
        elif self.provider == "ollama":
            self.model = settings.ollama_model
        else:
            raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'")

    # ---- model discovery: Groq retires models often, so never hard-code one ----
    def _pick_groq_model(self) -> str:
        try:
            available = {m.id for m in self.client.models.list().data}
        except Exception as e:  # network / auth problems surface here
            raise LLMError(f"Could not list Groq models: {e}") from e
        for m in settings.groq_models:
            if m in available:
                return m
        raise LLMError(f"None of {settings.groq_models} available. Your key sees: {sorted(available)}")

    # ---- core call ----
    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(4):
            try:
                if self.provider == "groq":
                    r = self.client.chat.completions.create(
                        model=self.model, messages=msgs,
                        temperature=temperature, max_tokens=max_tokens)
                    return r.choices[0].message.content or ""
                r = requests.post(f"{settings.ollama_url}/api/chat", timeout=300, json={
                    "model": self.model, "messages": msgs, "stream": False,
                    "options": {"temperature": temperature}})
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception as e:
                # free tier => rate limits (HTTP 429). Back off and retry.
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(2 ** attempt * 5)
                    continue
                raise LLMError(str(e)) from e
        raise LLMError("Rate-limited repeatedly. Wait a minute or switch model.")

    def chat_json(self, system: str, user: str) -> dict:
        """Ask for JSON and parse it defensively (works even without provider JSON-mode)."""
        txt = self.chat(system + "\nReturn ONLY valid JSON. No prose, no markdown fences.", user)
        return parse_json(txt)


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # reasoning models
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
