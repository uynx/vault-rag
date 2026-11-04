import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import NOTES, QUESTIONS  # noqa: E402


@pytest.fixture
def vault(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in NOTES.items():
        (corpus / f"{name}.md").write_text(text)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in QUESTIONS))
    return tmp_path
