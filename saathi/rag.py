"""Retrieval over the friend's own notes.

BM25 is always available (pure Python, no downloads, works on a plane).
If a local embedding model is reachable we fuse its ranking with BM25 using
reciprocal-rank fusion; if not, BM25 alone is surprisingly good on study notes.
"""
import math
import re
from collections import Counter

STOP = set(
    "the a an of to in and is are was were be been for on with as by at from that this "
    "it or which these those their its has have had not but can into than then so such "
    "also may will would should could each other more most some any all one two".split()
)

_SPLIT = re.compile(r"[\s\.,;:!?()\[\]{}\"'“”‘’।|/\\\-–—*_#>`]+")
_SENT = re.compile(r"(?<=[.!?।])\s+")


def tokenize(text: str):
    """Whitespace/punctuation tokenizer that keeps Devanagari words intact."""
    return [t for t in _SPLIT.split(text.lower()) if t]


def content_words(text: str):
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]


def sentences(text: str):
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT.split(flat) if s.strip()]


def chunk_text(text: str, max_words: int = 110, overlap: int = 15):
    """Split notes into study-sized chunks, respecting paragraph boundaries."""
    units = []
    for para in re.split(r"\n\s*\n", text.strip()):
        words = para.split()
        if not words:
            continue
        if len(words) <= max_words:
            units.append(" ".join(words))
        else:
            step = max_words - overlap
            for i in range(0, len(words), step):
                piece = words[i : i + max_words]
                units.append(" ".join(piece))
                if i + max_words >= len(words):
                    break
    chunks, cur = [], []
    for u in units:
        n = len(u.split())
        if cur and sum(len(c.split()) for c in cur) + n > max_words:
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(u)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


class BM25:
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 0.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query_tokens):
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            dl = len(d) or 1
            for t in query_tokens:
                f = self.tf[i].get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def _ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {i: r for r, i in enumerate(order)}


def search(query, texts, k=3, embeddings=None, query_emb=None):
    """Return [(index, score)] best-first. Only chunks with a real signal are returned."""
    if not texts:
        return []
    qtok = [t for t in tokenize(query) if t not in STOP]
    bm = BM25([tokenize(t) for t in texts]).scores(qtok)
    if embeddings and query_emb and len(embeddings) == len(texts) and all(embeddings):
        sem = [cosine(query_emb, e) for e in embeddings]
        rb, rs = _ranks(bm), _ranks(sem)
        fused = [1 / (60 + rb[i]) + 1 / (60 + rs[i]) for i in range(len(texts))]
        order = sorted(range(len(texts)), key=lambda i: fused[i], reverse=True)
        return [(i, fused[i]) for i in order[:k] if bm[i] > 0 or sem[i] > 0.3]
    order = sorted(range(len(texts)), key=lambda i: bm[i], reverse=True)
    return [(i, bm[i]) for i in order[:k] if bm[i] > 0]
