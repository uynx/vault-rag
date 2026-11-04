import hashlib
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLAUDE = ["claude", "-p", "--model", "sonnet", "--setting-sources", "project", "--disable-slash-commands",
          "--no-session-persistence", "--strict-mcp-config", "--tools", ""]


class CacheMiss(RuntimeError):
    pass


def claude_llm(prompt: str, cmd: list[str] = CLAUDE) -> str:
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=300)
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
