import pytest

from vault_rag.llm import CacheMiss, cached, claude_llm


def test_claude_llm_pipes_prompt_to_command_stdin():
    assert claude_llm("hello there", cmd=["cat"]) == "hello there"


def test_claude_llm_uses_once_ts_when_assistant_dir_is_set(tmp_path, monkeypatch):
    once = tmp_path / "once.ts"
    once.write_text("")
    monkeypatch.setenv("ASSISTANT_DIR", str(tmp_path))
    monkeypatch.delenv("VAULT_RAG_LLM_CMD", raising=False)
    from vault_rag.llm import default_cmd
    cmd = default_cmd()
    assert cmd[:2] == ["bun", str(once)]
    assert "-p" not in cmd


def test_claude_llm_passes_prompt_as_arg_for_once_ts(tmp_path):
    fake = tmp_path / "once.ts"
    fake.write_text("#!/bin/sh\necho \"$1\"\n")
    fake.chmod(0o755)
    assert claude_llm("hello once", cmd=[str(fake)]) == "hello once"


def test_claude_llm_raises_on_nonzero_exit():
    with pytest.raises(RuntimeError):
        claude_llm("x", cmd=["false"])


def test_cached_calls_underlying_once_and_persists(tmp_path):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return prompt.upper()

    path = tmp_path / "c.jsonl"
    a = cached(llm, path, "m1")
    assert a("abc") == "ABC" and a("abc") == "ABC"
    assert calls == ["abc"]
    b = cached(llm, path, "m1")
    assert b("abc") == "ABC" and calls == ["abc"]


def test_cache_key_includes_model(tmp_path):
    calls = []
    path = tmp_path / "c.jsonl"
    cached(lambda p: calls.append(1) or "a", path, "m1")("p")
    cached(lambda p: calls.append(2) or "b", path, "m2")("p")
    assert calls == [1, 2]


def test_offline_cache_miss_raises(tmp_path):
    f = cached(lambda p: "x", tmp_path / "c.jsonl", "m", offline=True)
    with pytest.raises(CacheMiss):
        f("never seen")
