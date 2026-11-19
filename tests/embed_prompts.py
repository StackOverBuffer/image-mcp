# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Regenerate the EMBEDDED PROMPTS block in server.py from the prompts/ files.

Run after changing any file under prompts/:

    uv run tests/embed_prompts.py

The block between the BEGIN/END markers is rewritten; nothing else in
server.py is touched. JSON files are validated while embedding.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server.py"
PROMPTS = ROOT / "prompts"

FILES = [
    "t2i/system_prompt.txt",
    "t2i/task_prompt.txt",
    "t2i/request_params.json",
    "edit/task_prompt.txt",
    "edit/request_params.json",
]

BEGIN = "# BEGIN EMBEDDED PROMPTS (generated block - do not edit)"
END = "# END EMBEDDED PROMPTS"


def build_block() -> str:
    entries = []
    for rel in FILES:
        path = PROMPTS / rel
        text = path.read_text(encoding="utf-8")
        if rel.endswith(".json"):
            json.loads(text)  # validate while embedding
        entries.append(f"    {rel!r}: {text!r},")
    return "EMBEDDED_PROMPTS: dict[str, str] = {\n" + "\n".join(entries) + "\n}"


def main() -> None:
    for rel in FILES:
        if not (PROMPTS / rel).is_file():
            raise SystemExit(f"missing source file: {PROMPTS / rel}")
    with open(SERVER, "r", encoding="utf-8", newline="") as f:
        src = f.read()
    if src.count(BEGIN) != 1 or src.count(END) != 1:
        raise SystemExit("BEGIN/END markers not found exactly once in server.py")
    start = src.index(BEGIN) + len(BEGIN)
    stop = src.index(END)
    new = src[:start] + "\n" + build_block() + "\n" + src[stop:]
    with open(SERVER, "w", encoding="utf-8", newline="") as f:
        f.write(new)
    total = sum(len((PROMPTS / rel).read_text(encoding="utf-8")) for rel in FILES)
    print(f"embedded {len(FILES)} prompt files ({total} chars) into {SERVER.name}")


if __name__ == "__main__":
    main()
