"""Spaced repetition: SM-2 with a confidence twist.

Plain SM-2 only knows right/wrong. Saathi also asks "were you sure?" because
being *confidently wrong* is the most useful signal a learner can give: those
are the misconceptions worth fixing first. So confident-wrong answers hit the
ease factor hardest and come back soonest.
"""

DAY = 86400
RELEARN_SECONDS = 10 * 60


def grade(correct: bool, confident: bool) -> int:
    """Map (correct, confident) to an SM-2 quality score 0..5."""
    if correct:
        return 5 if confident else 3
    return 0 if confident else 1


def schedule(ease: float, interval_days: float, reps: int, quality: int, now: int) -> dict:
    """Return the next card state. Pure function, easy to test."""
    ease = ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    ease = max(1.3, ease)
    if quality < 3:
        return {
            "ease": ease,
            "interval_days": 0.0,
            "reps": 0,
            "due": now + RELEARN_SECONDS,
            "lapsed": True,
        }
    reps += 1
    if reps == 1:
        interval = 1.0
    elif reps == 2:
        interval = 6.0
    else:
        interval = round(interval_days * ease, 1)
    if quality == 3:
        # got it right but guessed: don't stretch the gap as far
        interval = max(1.0, round(interval * 0.6, 1))
    return {
        "ease": ease,
        "interval_days": interval,
        "reps": reps,
        "due": now + int(interval * DAY),
        "lapsed": False,
    }
