"""Saathi: a local-first study companion. Flask API + static single-page UI."""
import hashlib
import hmac
import io
import json
import os
import random
import time
from collections import Counter

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

from . import observability, prompts, rag, srs, voice, web
from .db import DB
from .llm import extract_json, get_llm

MAX_NOTE_CHARS = 500_000


def qhash(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def validate_question(q, context):
    """Return a clean question dict, or None if it fails basic checks.

    The grounding check is the guard against a small model inventing facts: the
    correct option must share at least one word with the notes it was written from.
    """
    try:
        question = str(q["question"]).strip()
        options = [str(o).strip() for o in q["options"]]
        idx = int(q["answer_index"])
        explanation = str(q.get("explanation", "")).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if not question or len(options) != 4 or not all(options):
        return None
    if len({o.lower() for o in options}) != 4 or not 0 <= idx <= 3:
        return None
    ctx = set(rag.tokenize(context))
    ans = rag.tokenize(options[idx])
    if not ans or not any(t in ctx for t in ans):
        return None
    # shuffle deterministically: small models love putting the answer first
    rng = random.Random(qhash(question))
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    return {
        "question": question,
        "options": shuffled,
        "answer_index": order.index(idx),
        "explanation": explanation or "See your notes.",
    }


def create_app(db_path=None, llm=None):
    observability.init_sentry()  # no-op unless SENTRY_DSN is set
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.json.ensure_ascii = False

    if db_path is None:
        db_path = os.getenv("SAATHI_DB") or os.path.join(os.path.expanduser("~"), ".saathi", "saathi.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = DB(db_path)
    state = {"llm": observability.TracedLLM(llm or get_llm())}
    app.config["DB"] = db
    app.config["STATE"] = state

    def profile():
        return {**prompts.DEFAULT_PROFILE, **db.get_setting("profile", {})}

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    # ------------------------------------------------- access gate (cloud)
    # A public URL must not let strangers read notes or spend API credits. Set SAATHI_PASSWORD
    # on the server and the browser will ask once (any username, that password).
    password = os.getenv("SAATHI_PASSWORD", "")

    @app.before_request
    def gate():
        if not password or request.path == "/healthz":
            return None
        auth = request.authorization
        if auth and hmac.compare_digest((auth.password or "").encode(), password.encode()):
            return None
        return Response("Password needed.", 401, {"WWW-Authenticate": 'Basic realm="Saathi"'})

    @app.get("/healthz")
    def healthz():  # Render's health check: cheap, no DB, no model
        return "ok"

    # ------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        m = state["llm"]
        return jsonify(
            {
                "llm": m.kind,
                "model": m.name,
                "voice": voice.enabled(),
                "web": web.enabled(),
                "monitoring": observability.enabled(),
            }
        )

    # ---------------------------------------------------------- profile
    @app.get("/api/profile")
    def get_profile():
        return jsonify(profile())

    @app.post("/api/profile")
    def set_profile():
        data = request.get_json(silent=True) or {}
        clean = {k: str(data.get(k, "")).strip()[:300] for k in prompts.DEFAULT_PROFILE}
        if not clean["language_style"]:
            clean["language_style"] = prompts.DEFAULT_PROFILE["language_style"]
        if not clean["tone"]:
            clean["tone"] = prompts.DEFAULT_PROFILE["tone"]
        db.set_setting("profile", clean)
        return jsonify(clean)

    # ------------------------------------------------------------ notes
    @app.get("/api/notes")
    def list_notes():
        rows = db.rows(
            "SELECT d.id, d.title, d.created, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks, "
            "(SELECT COUNT(*) FROM cards k WHERE k.doc_id = d.id) AS cards "
            "FROM docs d ORDER BY d.id DESC"
        )
        return jsonify(rows)

    @app.post("/api/notes")
    def add_note():
        title, text = "", ""
        if request.files.get("file"):
            f = request.files["file"]
            title = request.form.get("title") or os.path.splitext(f.filename or "notes")[0]
            raw = f.read()
            if (f.filename or "").lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                except ImportError:
                    return err("PDF support needs `pip install pypdf`. Paste the text instead.")
                text = "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages)
            else:
                text = raw.decode("utf-8", errors="replace")
        else:
            data = request.get_json(silent=True) or {}
            title, text = str(data.get("title", "")).strip(), str(data.get("text", ""))
        text = text.strip()
        if len(text.split()) < 20:
            return err("These notes are too short to study from (need at least ~20 words).")
        if len(text) > MAX_NOTE_CHARS:
            return err("Notes are too long; please split them into smaller documents.")
        title = (title or text.split("\n", 1)[0][:60]).strip()
        doc_id = db.run("INSERT INTO docs(title, created) VALUES(?, ?)", (title, int(time.time())))
        pieces = rag.chunk_text(text)
        vecs = None
        try:
            vecs = state["llm"].embed(pieces)
        except Exception:
            vecs = None
        for i, piece in enumerate(pieces):
            emb = json.dumps(vecs[i]) if vecs and i < len(vecs) else None
            db.run(
                "INSERT INTO chunks(doc_id, idx, text, emb) VALUES(?,?,?,?)",
                (doc_id, i, piece, emb),
            )
        return jsonify({"id": doc_id, "title": title, "chunks": len(pieces), "embedded": bool(vecs)})

    @app.delete("/api/notes/<int:doc_id>")
    def delete_note(doc_id):
        db.run("DELETE FROM cards WHERE doc_id = ?", (doc_id,))
        db.run("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        db.run("DELETE FROM docs WHERE id = ?", (doc_id,))
        return jsonify({"ok": True})

    # --------------------------------------------------------- generate
    @app.post("/api/generate")
    def generate():
        data = request.get_json(silent=True) or {}
        doc_id = data.get("doc_id")
        max_chunks = max(1, min(int(data.get("max_chunks", 4)), 12))
        per_chunk = max(1, min(int(data.get("per_chunk", 2)), 4))
        sql = "SELECT * FROM chunks"
        args = []
        if doc_id:
            sql += " WHERE doc_id = ?"
            args.append(int(doc_id))
        sql += " ORDER BY gen_count ASC, id ASC LIMIT ?"
        args.append(max_chunks)
        chunks = db.rows(sql, tuple(args))
        if not chunks:
            return err("Add some notes first.")
        added = skipped = 0
        errors = []
        now = int(time.time())
        for ch in chunks:
            try:
                with observability.agent("Quiz Writer"):
                    reply = state["llm"].chat(
                        prompts.quiz_messages(profile(), ch["text"], per_chunk),
                        json_mode=True,
                        temperature=0.5,
                    )
                items = extract_json(reply).get("questions", [])
            except Exception as e:  # network, bad JSON, timeout: keep going
                errors.append(str(e)[:160])
                continue
            for q in items:
                v = validate_question(q, ch["text"])
                if not v:
                    skipped += 1
                    continue
                h = qhash(v["question"])
                if db.row("SELECT id FROM cards WHERE qhash = ?", (h,)):
                    skipped += 1
                    continue
                db.run(
                    "INSERT INTO cards(chunk_id, doc_id, qhash, question, options, answer_index,"
                    " explanation, due, created) VALUES(?,?,?,?,?,?,?,?,?)",
                    (ch["id"], ch["doc_id"], h, v["question"], json.dumps(v["options"], ensure_ascii=False),
                     v["answer_index"], v["explanation"], now, now),
                )
                added += 1
            db.run("UPDATE chunks SET gen_count = gen_count + 1 WHERE id = ?", (ch["id"],))
        if added == 0 and errors:
            return jsonify({"added": 0, "skipped": skipped, "errors": errors}), 502
        return jsonify({"added": added, "skipped": skipped, "errors": errors})

    # ------------------------------------------------------------ study
    @app.get("/api/next")
    def next_card():
        now = int(time.time())
        due_count = db.row("SELECT COUNT(*) AS n FROM cards WHERE due <= ?", (now,))["n"]
        card = db.row("SELECT * FROM cards WHERE due <= ? ORDER BY due ASC LIMIT 1", (now,))
        if not card:
            nxt = db.row("SELECT MIN(due) AS d FROM cards")
            wait = (nxt["d"] - now) if nxt and nxt["d"] else None
            total = db.row("SELECT COUNT(*) AS n FROM cards")["n"]
            return jsonify({"card": None, "due": 0, "total": total, "next_due_in": wait})
        return jsonify(
            {
                "card": {
                    "id": card["id"],
                    "question": card["question"],
                    "options": json.loads(card["options"]),
                    "is_new": card["reps"] == 0 and card["lapses"] == 0,
                },
                "due": due_count,
            }
        )

    @app.post("/api/answer")
    def answer():
        data = request.get_json(silent=True) or {}
        try:
            card_id, chosen = int(data["card_id"]), int(data["chosen"])
        except (KeyError, TypeError, ValueError):
            return err("card_id and chosen are required")
        confident = bool(data.get("confident"))
        card = db.row("SELECT * FROM cards WHERE id = ?", (card_id,))
        if not card:
            return err("No such card", 404)
        options = json.loads(card["options"])
        if not 0 <= chosen < len(options):
            return err("chosen is out of range")
        now = int(time.time())
        correct = chosen == card["answer_index"]
        q = srs.grade(correct, confident)
        nxt = srs.schedule(card["ease"], card["interval_days"], card["reps"], q, now)
        db.run(
            "UPDATE cards SET ease=?, interval_days=?, reps=?, due=?, lapses=lapses+? WHERE id=?",
            (nxt["ease"], nxt["interval_days"], nxt["reps"], nxt["due"], 1 if nxt["lapsed"] else 0, card_id),
        )
        attempt_id = db.run(
            "INSERT INTO attempts(card_id, ts, chosen, correct, confident) VALUES(?,?,?,?,?)",
            (card_id, now, chosen, int(correct), int(confident)),
        )
        return jsonify(
            {
                "attempt_id": attempt_id,
                "correct": correct,
                "correct_index": card["answer_index"],
                "explanation": card["explanation"],
                "confident_but_wrong": confident and not correct,
                "next_review_days": nxt["interval_days"],
            }
        )

    @app.post("/api/attempts/<int:attempt_id>/diagnose")
    def diagnose(attempt_id):
        """Separate call so the answer feels instant; the 'why you slipped' arrives a moment later."""
        a = db.row("SELECT * FROM attempts WHERE id = ?", (attempt_id,))
        if not a:
            return err("No such attempt", 404)
        if a["correct"]:
            return jsonify({"tag": None, "why": None})
        if a["tag"]:
            return jsonify({"tag": a["tag"], "why": a["why"]})
        card = db.row("SELECT * FROM cards WHERE id = ?", (a["card_id"],))
        options = json.loads(card["options"])
        try:
            with observability.agent("Misconception Diagnoser"):
                reply = state["llm"].chat(
                    prompts.diagnose_messages(
                        profile(), card["question"], options, card["answer_index"], a["chosen"], card["explanation"]
                    ),
                    json_mode=True,
                )
            d = extract_json(reply)
            tag = str(d.get("tag", "")).strip().lower()[:40] or "unclear"
            why = str(d.get("why", "")).strip()[:400]
        except Exception:
            tag, why = "unclear", ""
        db.run("UPDATE attempts SET tag = ?, why = ? WHERE id = ?", (tag, why, attempt_id))
        return jsonify({"tag": tag, "why": why})

    # -------------------------------------------------------------- ask
    @app.post("/api/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        if not question:
            return err("Ask something first.")
        chunks = db.rows("SELECT c.id, c.text, c.emb, d.title FROM chunks c JOIN docs d ON d.id = c.doc_id")
        if not chunks:
            return err("Add some notes first.")
        embs = [json.loads(c["emb"]) if c["emb"] else None for c in chunks]
        qemb = None
        if all(embs):
            try:
                v = state["llm"].embed([question])
                qemb = v[0] if v else None
            except Exception:
                qemb = None
        hits = rag.search(question, [c["text"] for c in chunks], k=3, embeddings=embs, query_emb=qemb)
        if not hits:
            return jsonify({"answer": "I could not find that in your notes. Try adding a page that covers it.",
                            "sources": []})
        passages = [chunks[i]["text"] for i, _ in hits]
        web_hits = []
        if data.get("web") and web.enabled():  # only when the learner ticks "also search the web"
            try:
                web_hits = web.search(question)
            except (requests.RequestException, ValueError):
                web_hits = []
        try:
            with observability.agent("Notes Assistant"):
                reply = state["llm"].chat(
                    prompts.ask_messages(profile(), question, passages, [f"{w['title']}: {w['snippet']}" for w in web_hits])
                )
        except Exception as e:
            return err(f"The local model did not respond: {str(e)[:160]}", 502)
        sources = [{"n": n + 1, "title": chunks[i]["title"], "text": chunks[i]["text"][:240]}
                   for n, (i, _) in enumerate(hits)]
        sources += [{"n": len(hits) + k + 1, "title": w["title"], "text": w["snippet"], "url": w["url"]}
                    for k, w in enumerate(web_hits)]
        return jsonify({"answer": reply.strip(), "sources": sources})

    # ------------------------------------------------------------ voice
    @app.post("/api/speak")
    def speak():
        if not voice.enabled():
            return err("Voice is not set up on this server (needs ELEVENLABS_API_KEY).", 404)
        text = str((request.get_json(silent=True) or {}).get("text", "")).strip()
        if not text:
            return err("Nothing to read aloud.")
        try:
            audio = voice.synthesize(text)
        except requests.RequestException as e:
            return err(f"The voice service did not respond: {str(e)[:120]}", 502)
        return Response(audio, mimetype="audio/mpeg")

    # ------------------------------------------------------------ stats
    def compute_stats(now=None):
        now = now or int(time.time())
        week_ago = now - 7 * 86400
        att7 = db.rows("SELECT * FROM attempts WHERE ts >= ?", (week_ago,))
        days = {time.strftime("%Y-%m-%d", time.localtime(a["ts"])) for a in db.rows("SELECT ts FROM attempts")}
        streak, day = 0, now
        if time.strftime("%Y-%m-%d", time.localtime(day)) not in days:
            day -= 86400
        while time.strftime("%Y-%m-%d", time.localtime(day)) in days:
            streak += 1
            day -= 86400
        tags = Counter(a["tag"] for a in att7 if a["tag"] and a["tag"] != "unclear")
        examples = {}
        for a in att7:
            if a["tag"] in tags and a["tag"] not in examples:
                c = db.row("SELECT question FROM cards WHERE id = ?", (a["card_id"],))
                if c:
                    examples[a["tag"]] = c["question"][:140]
        topics = db.rows(
            "SELECT d.title, COUNT(a.id) AS n, SUM(a.correct) AS ok FROM attempts a "
            "JOIN cards c ON c.id = a.card_id JOIN docs d ON d.id = c.doc_id "
            "WHERE a.ts >= ? GROUP BY d.id HAVING n >= 3 ORDER BY (1.0*ok/n) DESC", (week_ago,))
        return {
            "total_cards": db.row("SELECT COUNT(*) AS n FROM cards")["n"],
            "due_now": db.row("SELECT COUNT(*) AS n FROM cards WHERE due <= ?", (now,))["n"],
            "mature_cards": db.row("SELECT COUNT(*) AS n FROM cards WHERE interval_days >= 6")["n"],
            "answered_7d": len(att7),
            "accuracy_7d": round(100 * sum(a["correct"] for a in att7) / len(att7)) if att7 else 0,
            "days_active_7d": len({time.strftime("%Y-%m-%d", time.localtime(a["ts"])) for a in att7}),
            "streak": streak,
            "confident_wrong_7d": sum(1 for a in att7 if a["confident"] and not a["correct"]),
            "top_misconceptions": [
                {"tag": t, "count": n, "example_question": examples.get(t, "")} for t, n in tags.most_common(3)
            ],
            "best_topic": topics[0]["title"] if topics else None,
        }

    @app.get("/api/stats")
    def stats():
        return jsonify(compute_stats())

    @app.post("/api/letter")
    def letter():
        facts = compute_stats()
        facts["friend_name"] = profile()["friend_name"]
        try:
            with observability.agent("Letter Writer"):
                text = state["llm"].chat(prompts.letter_messages(profile(), facts), temperature=0.6)
        except Exception as e:
            return err(f"The local model did not respond: {str(e)[:160]}", 502)
        return jsonify({"letter": text.strip(), "facts": facts})

    return app
