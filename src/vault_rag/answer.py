import re

INSTRUCTIONS = (
    "Answer the question using only the numbered passages from a private notes vault. "
    "Cite the passages you used as [n] after each claim. Be brief. "
    "If the passages do not contain the answer, say \"The notes do not say.\""
)


def build_prompt(question: str, chunks) -> str:
    passages = "\n\n".join(f"[{i}] ({c.note}) {c.text}" for i, c in enumerate(chunks, 1))
    return f"{INSTRUCTIONS}\n\nPassages:\n{passages}\n\nQuestion: {question}\nAnswer:"


def cited_indices(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


def answer(question: str, retriever, llm, method: str = "hybrid", k: int = 5) -> dict:
    chunks = retriever.search(question, method, k)
    text = llm(build_prompt(question, chunks))
    cited = list(dict.fromkeys(chunks[i - 1].note for i in cited_indices(text, len(chunks))))
    return {"answer": text, "cited_notes": cited, "retrieved_notes": list(dict.fromkeys(c.note for c in chunks))}
