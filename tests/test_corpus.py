from vault_rag.corpus import chunk_note, load_corpus

NOTE = """<article>
    <header>
# Tide Mill
A mill on the estuary.
    </header>

    <section id="a">
## Grinding
* The mill grinds 40 kg of barley an hour ([[barley|the barley note]]).
* Flour goes to [[bakery]].
    </section>
</article>

**Parent**: [[hub]]
"""


def test_html_and_wikilinks_are_stripped():
    text = " ".join(c.text for c in chunk_note("tide_mill", NOTE))
    assert "<" not in text and "[[" not in text
    assert "the barley note" in text and "bakery" in text


def test_chunk_carries_note_id_and_title():
    chunks = chunk_note("tide_mill", NOTE)
    assert all(c.note == "tide_mill" for c in chunks)
    assert all("Tide Mill" in c.text for c in chunks)
    assert any("40 kg of barley" in c.text for c in chunks)


def test_long_section_is_split_with_overlap():
    words = " ".join(f"w{i}" for i in range(50))
    note = f"# Long\n## Part\n{words}\n"
    chunks = chunk_note("long", note, max_words=20, overlap=5)
    bodies = [c.text.split("Long ", 1)[-1].split() for c in chunks]
    assert len(chunks) == 3
    assert all(len(b) <= 20 + 4 for b in bodies)
    assert set(bodies[0][-5:]) & set(bodies[1])
    assert "w49" in chunks[-1].text


def test_load_corpus_uses_file_stem_as_note_id(tmp_path):
    (tmp_path / "one.md").write_text(NOTE)
    (tmp_path / "skip.txt").write_text("ignored")
    chunks = load_corpus(tmp_path)
    assert {c.note for c in chunks} == {"one"}


def test_tagline_under_the_title_shares_a_chunk_with_the_first_section():
    chunks = chunk_note("tide_mill", NOTE)
    assert len(chunks) == 1
    assert "A mill on the estuary." in chunks[0].text and "40 kg of barley" in chunks[0].text


def test_note_with_only_a_tagline_still_yields_a_chunk():
    assert [c.text for c in chunk_note("stub", "# Stub\nJust a tagline.\n")] == ["Stub\nJust a tagline."]
