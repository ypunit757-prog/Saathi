from conftest import NOTES


def test_health_reports_mock(client):
    assert client.get("/api/health").get_json()["llm"] == "mock"


def test_profile_roundtrip_with_defaults(client):
    r = client.post("/api/profile", json={"friend_name": "Riya", "goal": "GATE 2027", "language_style": ""})
    p = r.get_json()
    assert p["friend_name"] == "Riya" and p["language_style"]
    assert client.get("/api/profile").get_json()["goal"] == "GATE 2027"


def test_rejects_tiny_notes(client):
    assert client.post("/api/notes", json={"text": "too short"}).status_code == 400


def test_generate_requires_notes(client):
    assert client.post("/api/generate", json={}).status_code == 400


def _truth(c, card_id):
    """Look up the real answer straight from the DB (the API never leaks it)."""
    return c.application.config["DB"].row("SELECT answer_index FROM cards WHERE id=?", (card_id,))["answer_index"]


def _answer(c, right, confident=True):
    card = c.get("/api/next").get_json()["card"]
    truth = _truth(c, card["id"])
    chosen = truth if right else (truth + 1) % 4
    return c.post("/api/answer", json={"card_id": card["id"], "chosen": chosen, "confident": confident}).get_json()


def test_next_never_leaks_the_answer(loaded):
    card = loaded.get("/api/next").get_json()["card"]
    assert card and len(card["options"]) == 4
    assert "answer_index" not in card and "explanation" not in card


def test_right_answer_is_pushed_out_and_wrong_comes_back_in_minutes(loaded):
    ok = _answer(loaded, right=True)
    assert ok["correct"] and ok["next_review_days"] >= 1 and not ok["confident_but_wrong"]
    bad = _answer(loaded, right=False)
    assert not bad["correct"] and bad["next_review_days"] == 0 and bad["confident_but_wrong"]


def test_answered_card_leaves_the_due_queue(loaded):
    before = loaded.get("/api/stats").get_json()["due_now"]
    _answer(loaded, right=True)
    assert loaded.get("/api/stats").get_json()["due_now"] == before - 1


def test_stats_track_streak_accuracy_and_confident_mistakes(loaded):
    _answer(loaded, right=True)
    _answer(loaded, right=False, confident=True)
    s = loaded.get("/api/stats").get_json()
    assert s["answered_7d"] == 2 and s["accuracy_7d"] == 50
    assert s["confident_wrong_7d"] == 1 and s["streak"] == 1


def test_diagnose_wrong_answer_is_cached_and_feeds_stats(loaded):
    r = _answer(loaded, right=False)
    d = loaded.post(f"/api/attempts/{r['attempt_id']}/diagnose").get_json()
    assert d["tag"] and d["why"]
    assert loaded.post(f"/api/attempts/{r['attempt_id']}/diagnose").get_json() == d
    tops = loaded.get("/api/stats").get_json()["top_misconceptions"]
    assert tops and tops[0]["tag"] == d["tag"] and tops[0]["example_question"]


def test_diagnose_right_answer_returns_nothing(loaded):
    r = _answer(loaded, right=True)
    assert loaded.post(f"/api/attempts/{r['attempt_id']}/diagnose").get_json() == {"tag": None, "why": None}


def test_answer_validation(loaded):
    assert loaded.post("/api/answer", json={}).status_code == 400
    assert loaded.post("/api/answer", json={"card_id": 99999, "chosen": 0}).status_code == 404
    card = loaded.get("/api/next").get_json()["card"]
    assert loaded.post("/api/answer", json={"card_id": card["id"], "chosen": 9}).status_code == 400


def test_generate_is_idempotent_on_duplicates(loaded):
    before = loaded.get("/api/stats").get_json()["total_cards"]
    r = loaded.post("/api/generate", json={"max_chunks": 6, "per_chunk": 3}).get_json()
    after = loaded.get("/api/stats").get_json()["total_cards"]
    assert after == before + r["added"]
    assert r["skipped"] >= 1  # mock is deterministic, so repeats must be skipped


def test_ask_cites_notes_and_admits_ignorance(loaded):
    good = loaded.post("/api/ask", json={"question": "Where does the Calvin cycle occur?"}).get_json()
    assert "stroma" in good["answer"].lower() and good["sources"]
    bad = loaded.post("/api/ask", json={"question": "Who won the 1966 football world cup?"}).get_json()
    assert bad["sources"] == [] and "could not find" in bad["answer"].lower()


def test_letter_uses_only_real_facts(loaded):
    c = loaded
    c.post("/api/profile", json={"friend_name": "Riya"})
    _answer(c, right=True)
    out = c.post("/api/letter").get_json()
    assert out["facts"]["answered_7d"] == 1
    assert "Riya" in out["letter"] and "1 questions" in out["letter"]


def test_empty_letter_is_honest(client):
    out = client.post("/api/letter").get_json()
    assert out["facts"]["answered_7d"] == 0 and "quiet" in out["letter"]


def test_delete_note_removes_cards(loaded):
    doc = loaded.get("/api/notes").get_json()[0]
    assert doc["cards"] > 0
    loaded.delete(f"/api/notes/{doc['id']}")
    assert loaded.get("/api/notes").get_json() == []
    assert loaded.get("/api/stats").get_json()["total_cards"] == 0


def test_file_upload_text(client):
    import io
    r = client.post("/api/notes", data={"file": (io.BytesIO(NOTES.encode()), "bio.md")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["title"] == "bio"


class _BrokenLLM:
    kind, name = "broken", "broken"

    def __init__(self, reply=None):
        self.reply = reply

    def embed(self, texts):
        return None

    def chat(self, messages, json_mode=False, temperature=0.3):
        if self.reply is None:
            raise ConnectionError("model is down")
        return self.reply


def _app_with(tmp_path, llm):
    from saathi.app import create_app
    c = create_app(db_path=str(tmp_path / "x.db"), llm=llm).test_client()
    c.post("/api/notes", json={"title": "T", "text": NOTES})
    return c


def test_model_down_gives_clear_502_not_a_crash(tmp_path):
    c = _app_with(tmp_path, _BrokenLLM())
    r = c.post("/api/generate", json={})
    assert r.status_code == 502 and r.get_json()["errors"]
    assert c.post("/api/ask", json={"question": "Calvin cycle stroma"}).status_code == 502
    assert c.post("/api/letter").status_code == 502


def test_garbage_model_output_adds_no_cards(tmp_path):
    c = _app_with(tmp_path, _BrokenLLM(reply="Sorry, I can't do that."))
    assert c.post("/api/generate", json={}).status_code == 502
    assert c.get("/api/stats").get_json()["total_cards"] == 0


def test_hallucinated_questions_are_filtered(tmp_path):
    bad = '{"questions":[{"question":"Capital of Mars?","options":["Olympus","Zorg","Blip","Quux"],"answer_index":1,"explanation":"x"}]}'
    c = _app_with(tmp_path, _BrokenLLM(reply=bad))
    r = c.post("/api/generate", json={}).get_json()
    assert r["added"] == 0 and r["skipped"] >= 1
