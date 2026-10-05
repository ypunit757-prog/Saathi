"""LLM backends.

OllamaLLM talks to a local Ollama server (open-weight models, no internet needed).
MockLLM is a deterministic stand-in so the whole app, the tests and the demo run on a
machine with no GPU and no model. It is deliberately simple, but it is *grounded*:
it builds cloze questions from the notes themselves.
"""
import hashlib
import json
import os
import random
import re
import threading

import requests

from .rag import STOP, content_words, sentences

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_EMBED = "nomic-embed-text"
# Hosted Gemma through the Gemini API (native generateContent; Google's OpenAI-compatible
# endpoint documents Gemini models only and answers 404 for Gemma). Any OpenAI-compatible
# provider (DigitalOcean Gradient, a GPU Droplet running Ollama/vLLM, ...) works with
# SAATHI_CLOUD_PROVIDER=openai plus SAATHI_CLOUD_BASE_URL.
DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_CLOUD_MODEL = "gemma-4-31b-it"


def _http_error(r):
    """requests' default message hides the server's explanation; include it."""
    try:
        detail = r.json().get("error", {}).get("message") or r.text
    except ValueError:
        detail = r.text
    raise requests.HTTPError(f"{r.status_code} from {r.url.split('?')[0]}: {str(detail)[:300]}", response=r)


def extract_json(text: str):
    """Pull the first JSON object out of a model reply (handles ``` fences and chatter)."""
    if not text:
        raise ValueError("empty reply")
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object")


class OllamaLLM:
    kind = "ollama"

    def __init__(self, host=None, model=None, embed_model=None, timeout=180):
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = model or os.getenv("SAATHI_MODEL", DEFAULT_MODEL)
        self.embed_model = embed_model or os.getenv("SAATHI_EMBED_MODEL", DEFAULT_EMBED)
        self.timeout = timeout
        self._embed_ok = None

    @property
    def name(self):
        return self.model

    def available(self) -> bool:
        try:
            r = requests.get(self.host + "/api/tags", timeout=2)
            return r.ok
        except requests.RequestException:
            return False

    def chat(self, messages, json_mode=False, temperature=0.3):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(self.host + "/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def embed(self, texts):
        """Return list of vectors, or None if no embedding model is available."""
        if self._embed_ok is False:
            return None
        try:
            r = requests.post(
                self.host + "/api/embed",
                json={"model": self.embed_model, "input": texts},
                timeout=self.timeout,
            )
            r.raise_for_status()
            vecs = r.json().get("embeddings")
            self._embed_ok = bool(vecs)
            return vecs or None
        except requests.RequestException:
            self._embed_ok = False
            return None


class CloudLLM:
    """Any OpenAI-compatible chat API (SAATHI_CLOUD_PROVIDER=openai + SAATHI_CLOUD_BASE_URL)."""

    kind = "cloud"

    def __init__(self, base_url=None, api_key=None, model=None, timeout=120):
        self.base_url = (base_url or os.getenv("SAATHI_CLOUD_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("SAATHI_CLOUD_API_KEY", "")
        self.model = model or os.getenv("SAATHI_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)
        self.timeout = timeout
        self._local = threading.local()

    @property
    def name(self):
        return self.model

    @property
    def last_usage(self):
        return getattr(self._local, "usage", None)

    def available(self) -> bool:
        return bool(self.api_key)

    def _prepare(self, messages):
        # Gemma has no system role: fold the instructions into the first user turn.
        if not self.model.lower().startswith("gemma") or not messages or messages[0]["role"] != "system":
            return messages
        merged = messages[0]["content"] + "\n\n" + messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        return [{"role": "user", "content": merged}] + list(messages[2:])

    def chat(self, messages, json_mode=False, temperature=0.3):
        payload = {"model": self.model, "messages": self._prepare(messages), "temperature": temperature}
        if json_mode and os.getenv("SAATHI_CLOUD_JSON_MODE") == "1":  # not every model supports it
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(
            self.base_url + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if not r.ok:
            _http_error(r)
        data = r.json()
        self._local.usage = data.get("usage")
        return data["choices"][0]["message"]["content"] or ""

    def embed(self, texts):
        return None  # retrieval falls back to BM25, which needs no embedding model


class GeminiLLM:
    """Gemma (or Gemini) through the Gemini API's native generateContent endpoint."""

    kind = "cloud"

    def __init__(self, api_key=None, model=None, base_url=None, timeout=120):
        self.api_key = api_key or os.getenv("SAATHI_CLOUD_API_KEY", "")
        self.model = model or os.getenv("SAATHI_CLOUD_MODEL") or DEFAULT_CLOUD_MODEL
        self.base_url = (base_url or os.getenv("SAATHI_CLOUD_BASE_URL") or DEFAULT_GEMINI_BASE).rstrip("/")
        self.thinking = os.getenv("SAATHI_CLOUD_THINKING", "minimal")  # "" to leave the API default
        self.timeout = timeout
        self._local = threading.local()

    @property
    def name(self):
        return self.model

    @property
    def last_usage(self):
        return getattr(self._local, "usage", None)

    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _body(messages, temperature, json_mode, thinking):
        system = [m["content"] for m in messages if m["role"] == "system"]
        body = {
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
                if m["role"] != "system"
            ],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        if json_mode and os.getenv("SAATHI_CLOUD_JSON_MODE") == "1":
            body["generationConfig"]["responseMimeType"] = "application/json"
        if thinking:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking}
        return body

    def chat(self, messages, json_mode=False, temperature=0.3):
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        thinking = self.thinking
        for attempt in range(2):
            r = requests.post(url, json=self._body(messages, temperature, json_mode, thinking),
                              headers=headers, timeout=self.timeout)
            if r.status_code == 400 and thinking and attempt == 0:
                thinking = ""  # this model does not accept thinkingConfig: retry without it
                continue
            break
        if not r.ok:
            _http_error(r)
        data = r.json()
        meta = data.get("usageMetadata") or {}
        self._local.usage = {"prompt_tokens": meta.get("promptTokenCount"),
                             "completion_tokens": meta.get("candidatesTokenCount")}
        cands = data.get("candidates") or []
        if not cands:
            raise ValueError("the model returned no answer (" + str(data.get("promptFeedback", "blocked or empty")) + ")")
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def embed(self, texts):
        return None  # retrieval falls back to BM25, which needs no embedding model


class MockLLM:
    kind = "mock"
    name = "mock (no model running)"

    def available(self):
        return True

    def embed(self, texts):
        return None

    # -- helpers ------------------------------------------------------
    @staticmethod
    def _between(text, start, end):
        m = re.search(re.escape(start) + r"\n?(.*?)" + re.escape(end), text, re.S)
        return m.group(1).strip() if m else ""

    def chat(self, messages, json_mode=False, temperature=0.3):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "[TASK:quiz]" in system:
            return json.dumps(self._quiz(user))
        if "[TASK:diagnose]" in system:
            return json.dumps(self._diagnose(user))
        if "[TASK:ask]" in system:
            return self._ask(user)
        if "[TASK:letter]" in system:
            return self._letter(user)
        return "I can only help with study tasks in mock mode."

    def _quiz(self, user):
        count = int((re.search(r"COUNT:\s*(\d+)", user) or [0, 3])[1])
        ctx = self._between(user, "CONTEXT:", "END_CONTEXT")
        ctx = "\n".join(l for l in ctx.splitlines() if not l.lstrip().startswith("#"))  # drop markdown headings
        sents = [s for s in sentences(ctx) if len(s.split()) >= 8]
        pool = []
        for s in sents:
            for w in re.findall(r"[A-Za-z][A-Za-z\-]{5,}", s):
                if w.lower() not in STOP and w.lower() not in pool:
                    pool.append(w.lower())
        out = []
        for s in sents:
            cands = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{5,}", s) if w.lower() not in STOP]
            if not cands:
                continue
            answer = max(cands, key=len)
            distractors = [w for w in pool if w != answer.lower() and abs(len(w) - len(answer)) < 6]
            if len(distractors) < 3:
                continue
            rng = random.Random(hashlib.md5(s.encode()).hexdigest())
            opts = rng.sample(distractors, 3) + [answer.lower()]
            rng.shuffle(opts)
            blank = re.sub(re.escape(answer), "_____", s, count=1, flags=re.I)
            out.append(
                {
                    "question": f"Fill in the blank from your notes: {blank}",
                    "options": opts,
                    "answer_index": opts.index(answer.lower()),
                    "explanation": f"Your notes say: {s}",
                }
            )
            if len(out) >= count:
                break
        return {"questions": out}

    def _diagnose(self, user):
        d = json.loads(user)
        return {
            "tag": "mixed up terms",
            "why": f"You picked '{d['learner_chose']}', which sits close to '{d['correct']}' "
            "in your notes, so the two may be blurring together.",
        }

    def _ask(self, user):
        q = self._between(user, "QUESTION:", "CONTEXT:") or user
        ctx = self._between(user, "CONTEXT:", "END_CONTEXT")
        qw = set(content_words(q))
        best, best_score = None, 0
        for s in sentences(ctx):
            score = len(qw & set(content_words(s)))
            if score > best_score:
                best, best_score = s, score
        if not best:
            return "I could not find that in your notes. Try adding a page that covers it."
        return f"From your notes: {best} [1]"

    def _letter(self, user):
        f = json.loads(self._between(user, "FACTS:", "END_FACTS"))
        name = f.get("friend_name") or "friend"
        bits = [f"Dear {name},", ""]
        if f.get("answered_7d"):
            bits.append(
                f"This week you answered {f['answered_7d']} questions and got "
                f"{f['accuracy_7d']}% right across {f['days_active_7d']} study days."
            )
        else:
            bits.append("This week was quiet, and that is okay. Ten minutes tomorrow is enough.")
        if f.get("top_misconceptions"):
            m = f["top_misconceptions"][0]
            bits.append(f"The one to fix first: '{m['tag']}', which came up {m['count']} times.")
        bits.append("Tomorrow: just do your due cards, nothing more.")
        bits += ["", "Saathi"]
        return "\n".join(bits)


def _cloud():
    """SAATHI_CLOUD_PROVIDER: 'gemini' (default, Gemma on the Gemini API) or 'openai' (any compatible API)."""
    if os.getenv("SAATHI_CLOUD_PROVIDER", "gemini").lower() == "openai":
        return CloudLLM()
    return GeminiLLM()


def get_llm(mode=None):
    """mode: 'ollama' | 'cloud' | 'mock' | 'auto' (default, env SAATHI_LLM).

    auto = a reachable local Ollama first (private), then a hosted model if
    SAATHI_CLOUD_API_KEY is set (this is what happens on Render), else the offline demo model.
    """
    mode = (mode or os.getenv("SAATHI_LLM", "auto")).lower()
    if mode == "mock":
        return MockLLM()
    if mode == "cloud":
        return _cloud()
    ollama = OllamaLLM()
    if mode == "ollama" or ollama.available():
        return ollama
    cloud = _cloud()
    if cloud.available():
        return cloud
    return MockLLM()
