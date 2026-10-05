"""Sentry monitoring + AI Agent tracing (optional: does nothing unless SENTRY_DSN is set).

Span layout follows Sentry's agent-tracing convention so the Explore > Agents views work:

    invoke_agent Saathi Quiz Writer      (op gen_ai.invoke_agent)
        chat gemma-3-27b-it              (op gen_ai.chat)

By default only numbers are recorded (model, tokens, latency). Prompts and notes are recorded
only if you set SENTRY_CAPTURE_PROMPTS=1, because they are a friend's private study notes.
"""
import contextlib
import contextvars
import json
import os
import re
import sys
import time

try:  # sentry-sdk is optional
    import sentry_sdk
except ImportError:  # pragma: no cover
    sentry_sdk = None

_TASK = re.compile(r"\[TASK:(\w+)\]")
_enabled = False
_capture = False
_agent = contextvars.ContextVar("saathi_agent", default=None)


def init_sentry() -> bool:
    global _enabled, _capture
    dsn = os.getenv("SENTRY_DSN")
    if not dsn or sentry_sdk is None:
        return False
    _capture = os.getenv("SENTRY_CAPTURE_PROMPTS") == "1"
    opts = dict(
        dsn=dsn,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        send_default_pii=_capture,  # Sentry treats LLM inputs/outputs as PII: off unless opted in
        environment=os.getenv("SENTRY_ENVIRONMENT") or ("production" if os.getenv("RENDER") else "local"),
        release=os.getenv("RENDER_GIT_COMMIT"),
    )
    if os.getenv("SENTRY_SELF_HOSTED") == "1":  # SDK >= 2.64: self-hosted may not ingest standalone gen_ai spans
        opts["stream_gen_ai_spans"] = False
    sentry_sdk.init(**opts)
    _enabled = True
    return True


def enabled() -> bool:
    return _enabled


def _task_of(messages) -> str:
    m = _TASK.search(messages[0].get("content", "")) if messages else None
    return m.group(1) if m else "chat"


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _set(span, key, value):
    """Set a span attribute; telemetry must never be able to break the app."""
    if span is None:
        return
    try:
        (getattr(span, "set_attribute", None) or span.set_data)(key, value)
    except Exception:
        pass


@contextlib.contextmanager
def _span(op, name, attrs):
    """Open a Sentry span (yields None if Sentry misbehaves). Errors from the body still propagate."""
    try:
        cm = sentry_sdk.start_span(op=op, name=name)  # no attributes= kwarg: not supported by sentry-sdk 2.x
        span = cm.__enter__()
        for k, v in attrs.items():
            _set(span, k, v)
    except Exception:
        yield None
        return
    try:
        yield span
    except BaseException:
        try:
            cm.__exit__(*sys.exc_info())
        except Exception:
            pass
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


@contextlib.contextmanager
def agent(name: str):
    """One agent run, e.g. `with agent("Quiz Writer"):`. Model calls inside nest under it."""
    if not _enabled:
        yield
        return
    full = f"Saathi {name}"
    token = _agent.set(full)
    try:
        with _span(
            "gen_ai.invoke_agent",
            f"invoke_agent {full}",
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": full},
        ):
            yield
    finally:
        _agent.reset(token)


class TracedLLM:
    """Wraps any Saathi LLM backend and records each chat call as a Sentry `gen_ai.chat` span."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):  # kind, name, embed, available ...
        return getattr(self._inner, item)

    def chat(self, messages, json_mode=False, temperature=0.3):
        if not _enabled:
            return self._inner.chat(messages, json_mode=json_mode, temperature=temperature)
        model = self._inner.name
        attrs = {  # required attributes go on at span start so sampling can use them
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": self._inner.kind,
            "gen_ai.request.model": model,
            "gen_ai.request.temperature": temperature,
            "saathi.task": _task_of(messages),
        }
        if _agent.get():
            attrs["gen_ai.agent.name"] = _agent.get()
        if _capture:
            attrs["gen_ai.input.messages"] = json.dumps(
                [{"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]} for m in messages]
            )
        t0 = time.perf_counter()
        with _span("gen_ai.chat", f"chat {model}", attrs) as span:
            try:
                reply = self._inner.chat(messages, json_mode=json_mode, temperature=temperature)
            except Exception as exc:
                _set(span, "error.type", type(exc).__name__)
                raise
            usage = getattr(self._inner, "last_usage", None) or {}
            tin = usage.get("prompt_tokens") or _approx_tokens("".join(m["content"] for m in messages))
            tout = usage.get("completion_tokens") or _approx_tokens(reply or "")
            _set(span, "gen_ai.response.model", model)
            _set(span, "gen_ai.usage.input_tokens", tin)
            _set(span, "gen_ai.usage.output_tokens", tout)
            _set(span, "gen_ai.usage.total_tokens", tin + tout)
            _set(span, "saathi.latency_ms", round((time.perf_counter() - t0) * 1000))
            if _capture:
                _set(span, "gen_ai.output.messages", json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": reply}]}]))
            return reply
