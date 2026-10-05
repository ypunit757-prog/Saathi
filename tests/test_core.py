from saathi import rag, srs
from saathi.app import validate_question
from saathi.llm import extract_json


# ---------------------------------------------------------------- SM-2
def test_grade_mapping():
    assert srs.grade(True, True) == 5
    assert srs.grade(True, False) == 3
    assert srs.grade(False, False) == 1
    assert srs.grade(False, True) == 0


def test_interval_growth_and_lapse():
    s = srs.schedule(2.5, 0, 0, 5, now=1000)
    assert s["interval_days"] == 1 and s["reps"] == 1
    s = srs.schedule(s["ease"], s["interval_days"], s["reps"], 5, now=1000)
    assert s["interval_days"] == 6 and s["reps"] == 2
    s = srs.schedule(s["ease"], s["interval_days"], s["reps"], 5, now=1000)
    assert s["interval_days"] > 6
    lapsed = srs.schedule(s["ease"], s["interval_days"], s["reps"], 0, now=1000)
    assert lapsed["lapsed"] and lapsed["reps"] == 0 and lapsed["due"] == 1000 + srs.RELEARN_SECONDS


def test_confident_wrong_hurts_ease_more_than_unsure_wrong():
    sure = srs.schedule(2.5, 6, 2, 0, now=0)["ease"]
    unsure = srs.schedule(2.5, 6, 2, 1, now=0)["ease"]
    assert sure < unsure
    assert srs.schedule(1.3, 6, 2, 0, now=0)["ease"] == 1.3  # floor


def test_guessed_right_answer_gets_shorter_gap():
    sure = srs.schedule(2.5, 6, 2, 5, now=0)["interval_days"]
    guess = srs.schedule(2.5, 6, 2, 3, now=0)["interval_days"]
    assert guess < sure


# ----------------------------------------------------------------- RAG
def test_tokenize_keeps_devanagari_words_whole():
    assert rag.tokenize("प्रकाश संश्लेषण, क्या है?") == ["प्रकाश", "संश्लेषण", "क्या", "है"]


def test_chunking_respects_size_and_keeps_all_text():
    para = " ".join(f"word{i}" for i in range(300))
    chunks = rag.chunk_text(para, max_words=100, overlap=10)
    assert len(chunks) >= 3 and all(len(c.split()) <= 100 for c in chunks)
    assert "word0" in chunks[0] and "word299" in chunks[-1]


def test_bm25_finds_the_right_chunk():
    texts = [
        "The Calvin cycle fixes carbon dioxide in the stroma using RuBisCO.",
        "Mitochondria produce ATP through oxidative phosphorylation.",
        "The French Revolution began in 1789.",
    ]
    hits = rag.search("which enzyme fixes carbon dioxide", texts, k=2)
    assert hits and hits[0][0] == 0


def test_search_returns_nothing_for_unrelated_query():
    assert rag.search("quantum chromodynamics", ["plants make sugar from light"], k=3) == []


def test_embedding_fusion_prefers_semantic_match():
    texts = ["alpha beta", "gamma delta"]
    embs = [[1.0, 0.0], [0.0, 1.0]]
    hits = rag.search("alpha", texts, k=2, embeddings=embs, query_emb=[1.0, 0.0])
    assert hits[0][0] == 0


# ------------------------------------------------------- JSON handling
def test_extract_json_handles_fences_and_chatter():
    raw = 'Sure! Here you go:\n```json\n{"a": {"b": "x } y"}, "c": 1}\n```\nHope that helps'
    assert extract_json(raw) == {"a": {"b": "x } y"}, "c": 1}


def test_extract_json_rejects_garbage():
    for bad in ("", "no json here", '{"unterminated": '):
        try:
            extract_json(bad)
        except ValueError:
            continue
        raise AssertionError(f"should have failed: {bad!r}")


# --------------------------------------------------------- validation
CTX = "RuBisCO is the enzyme that catalyses carbon fixation in the Calvin cycle."


def _q(**kw):
    q = {"question": "Which enzyme starts carbon fixation?",
         "options": ["RuBisCO", "Amylase", "Lipase", "Pepsin"], "answer_index": 0, "explanation": "See notes."}
    q.update(kw)
    return q


def test_validate_accepts_grounded_question_and_keeps_answer_correct():
    v = validate_question(_q(), CTX)
    assert v and v["options"][v["answer_index"]] == "RuBisCO"


def test_validate_rejects_ungrounded_answer():
    assert validate_question(_q(options=["Telomerase", "Amylase", "Lipase", "Pepsin"]), CTX) is None


def test_validate_rejects_bad_shapes():
    assert validate_question(_q(options=["a", "a", "b", "c"]), CTX) is None
    assert validate_question(_q(options=["RuBisCO", "x", "y"]), CTX) is None
    assert validate_question(_q(answer_index=7), CTX) is None
    assert validate_question({"question": "?"}, CTX) is None
