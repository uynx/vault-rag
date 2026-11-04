import re
from dataclasses import dataclass
from pathlib import Path

TAG = re.compile(r"</?(?:article|header|section)[^>]*>")
WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
FOOTER = re.compile(r"^\*\*(?:Parent|Last Milestone Synthesis)\*\*")


@dataclass(frozen=True)
class Chunk:
    note: str
    idx: int
    text: str


def _clean(raw: str) -> list[str]:
    raw = WIKILINK.sub(r"\1", TAG.sub("", raw))
    return [ln.strip() for ln in raw.splitlines() if ln.strip() and not FOOTER.match(ln.strip())]


def _sections(lines: list[str]) -> tuple[str, list[tuple[str, list[str]]]]:
    title, sections, head, body = "", [], "", []
    for ln in lines:
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
        elif ln.startswith("## "):
            sections.append((head, body))
            head, body = ln[3:].strip(), []
        else:
            body.append(ln)
    sections.append((head, body))
    return title, [(h, b) for h, b in sections if b]


def _windows(words: list[str], size: int, overlap: int):
    start = 0
    while True:
        yield words[start : start + size]
        if start + size >= len(words):
            return
        start += size - overlap


def chunk_note(note: str, raw: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    title, sections = _sections(_clean(raw))
    chunks = []
    for head, body in sections:
        label = f"{title} / {head}" if head else title
        for window in _windows(" ".join(body).split(), max_words, overlap):
            chunks.append(Chunk(note, len(chunks), f"{label}\n{' '.join(window)}"))
    return chunks


def load_corpus(root: Path, exclude: tuple[str, ...] = (), **kw) -> list[Chunk]:
    paths = [p for p in sorted(Path(root).rglob("*.md")) if not set(p.relative_to(root).parts[:-1]) & set(exclude)]
    return [c for p in paths for c in chunk_note(p.stem, p.read_text(), **kw)]
