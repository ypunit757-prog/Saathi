# Saathi

An offline study companion, built for one person.

Saathi turns a friend's **own notes** into quiz questions, asks how sure they are before they answer, works out *why* a wrong answer was wrong, schedules each question for the moment they are about to forget it, and writes a short weekly letter in their own style of language. It runs an open-weight model (Gemma by default) through [Ollama](https://ollama.com) on a laptop. Nothing is sent anywhere.

> Built for the DEV Hacktoberfest Weekend Challenge, "Build for a Friend" (Oct 2 to 5, 2026). All code in this repo was started inside that window.

## Why it is shaped this way

- **Their notes, not the internet.** Questions are written only from the notes you add, and every generated question must pass a grounding check (the right answer has to appear in the source passage) before it is saved. Small models make things up; this keeps the damage out of the study deck.
- **Confidence matters.** Before answering, you say *Sure* or *Not sure*. A wrong answer you felt sure about is the most valuable signal there is, so it lowers that card's ease the most and is called out in the letter.
- **Mistakes get a reason.** After a wrong answer, the model is asked for the likely misunderstanding behind *that specific wrong choice* and tags it. Tags are counted over the week.
- **Works with the Wi-Fi off.** No CDN, no web fonts, no telemetry. The server binds to `127.0.0.1` only.
- **Works without a GPU or model too.** If no Ollama server is found, Saathi starts in a clearly labelled demo mode with a deterministic stand-in model that builds fill-in-the-blank questions from the notes. This is what the tests use.

## Deploy it (optional cloud mode)

Saathi can also run as a hosted web app on Render: see **[DEPLOY.md](DEPLOY.md)**. In that mode notes are sent to the hosted model provider (Gemma on Google AI by default), so the "nothing leaves the machine" promise above applies to local mode only. Optional integrations, each off unless its key is set: Sentry (error monitoring and AI call traces, no prompts or notes recorded), ElevenLabs (read the weekly letter aloud), SerpApi (opt-in web search in "Ask my notes"). A site password (`SAATHI_PASSWORD`) protects a public URL.

## Run it

```bash
# 1. a local model (one time)
ollama pull gemma3:4b
ollama pull nomic-embed-text      # optional: adds semantic search to "Ask my notes"

# 2. Saathi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m saathi                  # then open http://127.0.0.1:5000
```

On first launch Saathi asks who it is for (name, goal, how they like to be spoken to). Then add notes (paste, `.txt`, `.md` or `.pdf`), choose **Make questions**, and start studying. `sample_notes/photosynthesis.md` is there to try it out.

Settings (environment variables):

| variable | default | meaning |
|---|---|---|
| `SAATHI_MODEL` | `gemma3:4b` | any chat model you have pulled in Ollama |
| `SAATHI_EMBED_MODEL` | `nomic-embed-text` | embedding model for semantic search |
| `SAATHI_LLM` | `auto` | `ollama`, `cloud`, `mock`, or `auto` (Ollama if reachable, else the hosted model if a key is set, else demo) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | where Ollama listens |
| `SAATHI_DB` | `~/.saathi/saathi.db` | the one file holding everything |
| `SAATHI_PORT` | `5000` | web port |

Swapping models is one variable, which is the point: `SAATHI_MODEL=gemma3:1b python -m saathi` for an old laptop, a bigger model when you have the memory.

## Compare models on the real task

```bash
python bench/benchmark.py gemma3:1b gemma3:4b
```

Reports, per model, how often the reply was valid JSON, how many questions survived Saathi's validation, and seconds per chunk.

## How it works

| file | job |
|---|---|
| `saathi/llm.py` | Ollama client, offline mock model, tolerant JSON extraction |
| `saathi/rag.py` | chunking, BM25, optional embedding fusion (reciprocal rank) |
| `saathi/srs.py` | SM-2 spaced repetition with the confidence twist |
| `saathi/prompts.py` | every prompt, parameterised by the friend's profile |
| `saathi/observability.py` | optional Sentry setup and per-call AI spans |
| `saathi/voice.py`, `saathi/web.py` | optional ElevenLabs and SerpApi clients |
| `saathi/app.py` | Flask API: notes, question generation, study loop, diagnosis, ask, stats, letter |
| `saathi/static/` | the notebook-style interface (plain HTML, CSS, JS) |

Design notes: the answer to a card is never sent to the browser until you answer it; all model text is rendered with `textContent`; the diagnosis runs as a second request so the right/wrong feedback is instant even on a slow laptop.

## Tests

```bash
pytest
```

44 tests cover the cloud integrations (all mocked) plus the scheduler (including the confidence rules), retrieval, JSON extraction, question validation, and the whole study loop through the API.

## Limits, honestly

- Question quality depends on the model. 1B models write weaker distractors; the validation filter drops bad questions instead of fixing them, so you may see fewer questions than you asked for.
- The demo-mode questions are fill-in-the-blank only.
- Scanned PDFs with no text layer are not supported (no OCR).
- It is a single-user local app, with no accounts and no sync.
