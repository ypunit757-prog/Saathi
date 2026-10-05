"""Cloud/sponsor integrations: all external services are mocked, no network or keys needed."""
import json

import pytest

from conftest import NOTES
from saathi import observability, voice, web
from saathi.app import create_app
from saathi.llm import CloudLLM, MockLLM, get_llm


class FakeResp:
    def __init__(self, data=None, content=b"", ok=True):
        self._d, self.content, self.ok = data or {}, content, ok

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def make(tmp_path, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return create_app(db_path=str(tmp_path / "t.db"), llm=MockLLM()).test_client()


# ---- access gate + health check
def test_password_gate(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, SAATHI_PASSWORD="s3cret")
    assert c.get("/api/health").status_code == 401
    assert c.get("/healthz").status_code == 200  # Render must reach this without a password
    import base64
    ok = {"Authorization": "Basic " + base64.b64encode(b"me:s3cret").decode()}
    bad = {"Authorization": "Basic " + base64.b64encode(b"me:nope").decode()}
    assert c.get("/api/health", headers=ok).status_code == 200
    assert c.get("/api/health", headers=bad).status_code == 401


def test_no_password_means_open(client):
    assert client.get("/api/health").status_code == 200


# ---- hosted Gemma backend
def test_cloud_llm_folds_system_prompt_for_gemma(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, payload=json, headers=headers)
        return FakeResp({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})

    monkeypatch.setattr("saathi.llm.requests.post", fake_post)
    llm = CloudLLM(api_key="k", model="gemma-3-27b-it", base_url="https://x.test/v1")
    out = llm.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])
    assert out == "hi" and seen["url"] == "https://x.test/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert [m["role"] for m in seen["payload"]["messages"]] == ["user"]
    assert "SYS" in seen["payload"]["messages"][0]["content"] and "USER" in seen["payload"]["messages"][0]["content"]
    assert llm.last_usage["prompt_tokens"] == 7


def test_get_llm_cloud_fallback(monkeypatch):
    monkeypatch.setenv("SAATHI_LLM", "auto")
    monkeypatch.setattr("saathi.llm.OllamaLLM.available", lambda self: False)
    monkeypatch.delenv("SAATHI_CLOUD_API_KEY", raising=False)
    assert get_llm().kind == "mock"
    monkeypatch.setenv("SAATHI_CLOUD_API_KEY", "k")
    assert get_llm().kind == "cloud"


# ---- Sentry tracing wrapper
def test_traced_llm_passthrough_without_sentry(client):
    assert client.get("/api/health").get_json()["monitoring"] is False


def _fake_sentry(monkeypatch, capture=False):
    spans = []

    class Span:
        def __init__(self, **kw):
            self.kw, self.data = kw, dict(kw.get("attributes", {}))

        def set_data(self, k, v):
            self.data[k] = v

        def __enter__(self):
            spans.append(self)
            return self

        def __exit__(self, *a):
            return False

    class FakeSentry:
        start_span = staticmethod(lambda **kw: Span(**kw))

    monkeypatch.setattr(observability, "sentry_sdk", FakeSentry)
    monkeypatch.setattr(observability, "_enabled", True)
    monkeypatch.setattr(observability, "_capture", capture)
    return spans


def _call(llm):
    with observability.agent("Letter Writer"):
        llm.chat([{"role": "system", "content": "[TASK:letter] x"}, {"role": "user", "content": "FACTS:\n{}\nEND_FACTS"}])


def test_traced_llm_follows_sentry_agent_convention(monkeypatch):
    spans = _fake_sentry(monkeypatch)
    _call(observability.TracedLLM(MockLLM()))
    agent_span, chat = spans
    assert agent_span.kw["op"] == "gen_ai.invoke_agent" and agent_span.kw["name"] == "invoke_agent Saathi Letter Writer"
    assert agent_span.data["gen_ai.agent.name"] == "Saathi Letter Writer"
    assert chat.kw["op"] == "gen_ai.chat" and chat.kw["name"].startswith("chat ")
    for k in ("gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model", "gen_ai.agent.name"):
        assert k in chat.data
    assert chat.data["gen_ai.usage.input_tokens"] > 0 and chat.data["gen_ai.usage.total_tokens"] > 0
    # privacy: prompts are NOT attached unless SENTRY_CAPTURE_PROMPTS=1
    assert "gen_ai.input.messages" not in chat.data and "gen_ai.output.messages" not in chat.data


def test_traced_llm_can_capture_prompts_when_opted_in(monkeypatch):
    spans = _fake_sentry(monkeypatch, capture=True)
    _call(observability.TracedLLM(MockLLM()))
    chat = spans[1]
    assert json.loads(chat.data["gen_ai.input.messages"])[0]["parts"][0]["type"] == "text"
    assert "gen_ai.output.messages" in chat.data


# ---- ElevenLabs voice
def test_speak_disabled_without_key(client, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert client.post("/api/speak", json={"text": "hello"}).status_code == 404


def test_speak_returns_audio(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, ELEVENLABS_API_KEY="e")
    calls = {}

    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        calls.update(url=url, headers=headers, body=json)
        return FakeResp(content=b"ID3audio")

    monkeypatch.setattr(voice.requests, "post", fake_post)
    r = c.post("/api/speak", json={"text": "Dear Riya"})
    assert r.status_code == 200 and r.mimetype == "audio/mpeg" and r.data == b"ID3audio"
    assert calls["headers"]["xi-api-key"] == "e" and calls["body"]["text"] == "Dear Riya"
    assert c.get("/api/health").get_json()["voice"] is True
    assert c.post("/api/speak", json={"text": "  "}).status_code == 400


# ---- SerpApi web search
def test_ask_with_web_results(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, SERPAPI_API_KEY="s")
    c.post("/api/notes", json={"title": "Photosynthesis", "text": NOTES})

    def fake_get(url, params=None, timeout=None):
        return FakeResp({"organic_results": [
            {"title": "Chlorophyll", "snippet": "Chlorophyll absorbs light.", "link": "https://example.org/c"},
            {"title": "Bad", "snippet": "x", "link": "javascript:alert(1)"},
        ]})

    monkeypatch.setattr(web.requests, "get", fake_get)
    r = c.post("/api/ask", json={"question": "What does chlorophyll absorb?", "web": True}).get_json()
    urls = [s.get("url") for s in r["sources"]]
    assert "https://example.org/c" in urls and not any(u and u.startswith("javascript") for u in urls)
    # web is opt-in per question
    r2 = c.post("/api/ask", json={"question": "What does chlorophyll absorb?"}).get_json()
    assert not any("url" in s for s in r2["sources"])


def test_web_failure_does_not_break_ask(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, SERPAPI_API_KEY="s")
    c.post("/api/notes", json={"title": "P", "text": NOTES})
    import requests as rq

    def boom(*a, **k):
        raise rq.ConnectionError("down")

    monkeypatch.setattr(web.requests, "get", boom)
    r = c.post("/api/ask", json={"question": "What does chlorophyll absorb?", "web": True})
    assert r.status_code == 200
