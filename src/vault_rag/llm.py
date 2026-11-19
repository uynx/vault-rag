import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_ARGS = ["--model", "sonnet", "--setting-sources", "project", "--disable-slash-commands",
             "--no-session-persistence", "--strict-mcp-config", "--tools", ""]
CLAUDE = ["claude", "-p", *BASE_ARGS]


def default_cmd() -> list[str]:
    if override := os.environ.get("VAULT_RAG_LLM_CMD"):
        return shlex.split(override)
    if assistant_dir := os.environ.get("ASSISTANT_DIR"):
        once = Path(assistant_dir) / "once.ts"
        if once.exists():
            return ["bun", str(once), *BASE_ARGS]
    return list(CLAUDE)


class CacheMiss(RuntimeError):
    pass


def claude_llm(prompt: str, cmd: list[str] | None = None) -> str:
    cmd = default_cmd() if cmd is None else cmd
    as_arg = any(Path(part).name == "once.ts" for part in cmd)
    run_cmd = [*cmd, prompt] if as_arg else cmd
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(run_cmd, input=None if as_arg else prompt, capture_output=True, text=True, cwd=cwd, timeout=300)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}: {r.stderr[-300:]}")
    return r.stdout.strip()


def cached(llm, path: Path, model: str, offline: bool = False):
    path, lock, store = Path(path), threading.Lock(), {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            store[row["key"]] = row["out"]

    def call(prompt: str) -> str:
        key = hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()
        if key in store:
            return store[key]
        if offline:
            raise CacheMiss(f"no cached output for prompt starting {prompt[:60]!r}")
        out = llm(prompt)
        with lock:
            store[key] = out
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps({"key": key, "out": out}) + "\n")
        return out

    return call


def pmap(fn, items, workers: int = 8) -> list:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
