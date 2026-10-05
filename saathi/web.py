"""SerpApi live web search for "Ask my notes" (optional, and only when the learner asks for it)."""
import os

import requests

API = "https://serpapi.com/search.json"


def enabled() -> bool:
    return bool(os.getenv("SERPAPI_API_KEY"))


def search(query: str, n: int = 4):
    """Return [{title, snippet, url}] for the top web results (http/https links only)."""
    r = requests.get(
        API,
        params={"engine": "google", "q": query, "num": n, "api_key": os.environ["SERPAPI_API_KEY"]},
        timeout=20,
    )
    r.raise_for_status()
    out = []
    for item in r.json().get("organic_results", []):
        url = str(item.get("link", ""))
        snippet = str(item.get("snippet", "")).strip()
        if url.startswith(("http://", "https://")) and snippet:
            out.append({"title": str(item.get("title", url))[:120], "snippet": snippet[:400], "url": url})
        if len(out) >= n:
            break
    return out
