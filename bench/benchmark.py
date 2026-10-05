"""Compare local models on the job Saathi actually gives them: writing grounded quiz questions.

    python bench/benchmark.py gemma3:1b gemma3:4b

For each model it asks for 3 questions per chunk of the sample notes and reports:
  json_ok   replies that parsed as the requested JSON
  kept      questions that survived Saathi's validation (4 distinct options, grounded in the notes)
  sec/chunk average wall-clock time per chunk
Needs a running Ollama server and the models pulled. Prints a Markdown table.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saathi import prompts, rag  # noqa: E402
from saathi.app import validate_question  # noqa: E402
from saathi.llm import OllamaLLM, extract_json  # noqa: E402

NOTES = os.path.join(os.path.dirname(__file__), "..", "sample_notes", "photosynthesis.md")


def run(model, chunks, per_chunk=3):
    llm = OllamaLLM(model=model)
    json_ok = asked = kept = 0
    elapsed = 0.0
    for ch in chunks:
        t = time.time()
        try:
            reply = llm.chat(prompts.quiz_messages({}, ch, per_chunk), json_mode=True, temperature=0.5)
            items = extract_json(reply).get("questions", [])
            json_ok += 1
        except Exception as e:
            print(f"  [{model}] failed: {str(e)[:100]}", file=sys.stderr)
            items = []
        elapsed += time.time() - t
        asked += len(items)
        kept += sum(1 for q in items if validate_question(q, ch))
    return json_ok, len(chunks), asked, kept, elapsed / max(1, len(chunks))


def main():
    models = sys.argv[1:] or [os.getenv("SAATHI_MODEL", "gemma3:4b")]
    if not OllamaLLM().available():
        sys.exit("Ollama is not running. Start it with `ollama serve` and pull the models first.")
    chunks = rag.chunk_text(open(NOTES, encoding="utf-8").read(), max_words=60)
    print("| model | json_ok | questions asked | kept after validation | sec/chunk |")
    print("|---|---|---|---|---|")
    for m in models:
        ok, n, asked, kept, sec = run(m, chunks)
        print(f"| {m} | {ok}/{n} | {asked} | {kept} | {sec:.1f} |")


if __name__ == "__main__":
    main()
