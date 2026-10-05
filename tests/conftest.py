import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from saathi.app import create_app  # noqa: E402
from saathi.llm import MockLLM  # noqa: E402

NOTES = open(os.path.join(os.path.dirname(__file__), "..", "sample_notes", "photosynthesis.md"), encoding="utf-8").read()


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "t.db"), llm=MockLLM())
    return app.test_client()


@pytest.fixture
def loaded(client):
    r = client.post("/api/notes", json={"title": "Photosynthesis", "text": NOTES})
    assert r.status_code == 200
    g = client.post("/api/generate", json={"max_chunks": 6, "per_chunk": 3})
    assert g.status_code == 200 and g.get_json()["added"] > 0
    return client
