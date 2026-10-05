"""All prompts in one place. Each system prompt starts with a [TASK:...] marker
so the offline mock model can recognise what is being asked."""
import json

DEFAULT_PROFILE = {
    "friend_name": "",
    "goal": "",
    "language_style": "clear, simple English",
    "tone": "warm, encouraging, never preachy",
    "motivation": "",
}


def persona(profile: dict) -> str:
    p = {**DEFAULT_PROFILE, **(profile or {})}
    who = p["friend_name"] or "your friend"
    lines = [
        f"You are Saathi, a study companion built by one person for {who}.",
        f"Speak in this style: {p['language_style']}.",
        f"Tone: {p['tone']}.",
    ]
    if p["goal"]:
        lines.append(f"{who} is working toward: {p['goal']}.")
    return " ".join(lines)


def quiz_messages(profile, context: str, count: int):
    system = (
        "[TASK:quiz] " + persona(profile) + "\n"
        "You write multiple-choice questions STRICTLY from the notes provided. "
        "Never use outside facts. Each question must have exactly 4 distinct options "
        "and exactly one correct option. Prefer questions that test understanding "
        "(why/how/what happens if) over trivia. Wrong options should be plausible "
        "mistakes a learner could really make. The explanation is 1-2 sentences in "
        "the speaking style above and must point back to the notes.\n"
        'Reply with JSON only: {"questions":[{"question":str,"options":[str,str,str,str],'
        '"answer_index":0-3,"explanation":str}]}'
    )
    user = f"COUNT: {count}\nCONTEXT:\n{context}\nEND_CONTEXT"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def diagnose_messages(profile, question, options, correct_i, chosen_i, explanation):
    system = (
        "[TASK:diagnose] " + persona(profile) + "\n"
        "A learner picked a wrong answer. Work out the most likely misunderstanding "
        "behind THAT specific wrong choice. Do not just restate the right answer.\n"
        'Reply with JSON only: {"tag": a 2-4 word lowercase label for the misconception, '
        '"why": one kind sentence addressed to the learner as "you"}'
    )
    user = json.dumps(
        {
            "question": question,
            "options": options,
            "correct": options[correct_i],
            "learner_chose": options[chosen_i],
            "notes_explanation": explanation,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ask_messages(profile, question, passages, web_passages=()):
    allp = [*passages, *(f"(web result) {w}" for w in web_passages)]
    numbered = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(allp))
    system = (
        "[TASK:ask] " + persona(profile) + "\n"
        "Answer the question using ONLY the numbered passages from the learner's own notes. "
        "Cite passages like [1]. If the notes do not contain the answer, say so plainly and "
        "suggest what to add to the notes. Keep it under 150 words."
        + (" Passages marked (web result) come from a web search, not the notes: if you use one, "
           "say clearly that it is from the web." if web_passages else "")
    )
    user = f"QUESTION: {question}\nCONTEXT:\n{numbered}\nEND_CONTEXT"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def letter_messages(profile, facts: dict):
    p = {**DEFAULT_PROFILE, **(profile or {})}
    system = (
        "[TASK:letter] " + persona(profile) + "\n"
        "Write a short weekly letter (120-180 words) from Saathi to the learner. "
        "Use ONLY the facts given: do not invent scores, streaks or topics. Celebrate one "
        "real win, name the one or two misconceptions worth fixing first (confident mistakes "
        "matter most), and end with one small concrete step for tomorrow. "
        + (f"Something that keeps them going: {p['motivation']}. " if p["motivation"] else "")
        + "No bullet points. Sign off as Saathi."
    )
    user = "FACTS:\n" + json.dumps(facts, ensure_ascii=False, indent=1) + "\nEND_FACTS"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
