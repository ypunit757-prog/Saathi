"""ElevenLabs text-to-speech: Saathi reads the weekly letter aloud (optional)."""
import os

import requests

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # a stock ElevenLabs voice; set ELEVENLABS_VOICE_ID to change
DEFAULT_MODEL = "eleven_multilingual_v2"  # handles Hindi and Hinglish text
MAX_CHARS = 2500


def enabled() -> bool:
    return bool(os.getenv("ELEVENLABS_API_KEY"))


def synthesize(text: str) -> bytes:
    """Return MP3 bytes. Raises requests.RequestException on failure."""
    r = requests.post(
        API.format(voice=os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)),
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
        json={"text": text[:MAX_CHARS], "model_id": os.getenv("ELEVENLABS_MODEL", DEFAULT_MODEL)},
        timeout=60,
    )
    r.raise_for_status()
    return r.content
