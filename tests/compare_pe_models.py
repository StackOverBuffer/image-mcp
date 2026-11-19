# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Compare prompt-expansion models: same instructions through expand_prompt on
each model (timed), then generate from each model's first caption for a visual A/B.

Usage:  uv run tests/compare_pe_models.py            # default model list
        uv run tests/compare_pe_models.py modelA modelB
"""

import asyncio
import re
import struct
import sys
import time
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

INSTRUCTIONS = [
    "a lighthouse on a cliff at sunset",
    "a poster for a school science fair with title, date, and three experiment icons",
]


def png_dims(path: str) -> tuple[int, int]:
    data = Path(path).read_bytes()
    return struct.unpack(">II", data[16:24])


def parse_expansion(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    caption = lines[1] if len(lines) > 1 else ""
    m = re.search(r"(?:Recommended|Target) size:\s*\"?(\d+:\d+)", text)
    ratio = m.group(1) if m else ""
    return caption, ratio


async def main() -> None:
    from fastmcp import Client

    import server as srv

    models = sys.argv[1:] or ["qwen3.8-27b", "glm-5.3-flash"]
    print(f"comparing: {models}\n")
    results: dict[tuple[str, int], tuple[float, str]] = {}

    async with Client(srv.mcp) as client:
        for model in models:
            srv.PROMPT_LLM_MODEL = model  # per-run override of the module config
            for i, instr in enumerate(INSTRUCTIONS):
                t0 = time.perf_counter()
                res = await asyncio.wait_for(
                    client.call_tool("expand_prompt", {"instruction": instr}), timeout=1500
                )
                dt = time.perf_counter() - t0
                text = next(c for c in res.content if c.type == "text").text
                results[(model, i)] = (dt, text)
                print(f"{'=' * 20} {model} | instruction {i + 1} | expansion {dt:6.1f}s")
                print(text, "\n")

        for model in models:
            _, text = results[(model, 0)]
            caption, ratio = parse_expansion(text)
            t0 = time.perf_counter()
            res2 = await asyncio.wait_for(
                client.call_tool("generate_image", {"prompt": caption, "size": ratio}), timeout=700
            )
            dt2 = time.perf_counter() - t0
            texts = [c.text for c in res2.content if c.type == "text"]
            paths = [l.strip() for t in texts for l in t.splitlines() if l.strip().lower().endswith(".png")]
            if paths:
                w, h = png_dims(paths[0])
                print(f"{'=' * 20} generation from {model} caption | {dt2:6.1f}s | ratio {ratio} -> {w}x{h}")
                print(f"    {paths[0]}")
            else:
                print(f"{'=' * 20} generation from {model} caption FAILED:\n{texts}")

    print("\nCOMPARE COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
