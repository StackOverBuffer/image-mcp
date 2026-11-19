# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Live prompt-expansion check: expand a short idea with the official PE prompts,
then generate directly from the expansion at the recommended aspect ratio.

Usage:  uv run tests/live_expand_check.py "your short idea"
"""

import asyncio
import re
import struct
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))


def png_dims(path: str) -> tuple[int, int]:
    data = Path(path).read_bytes()
    return struct.unpack(">II", data[16:24])


async def main() -> None:
    from fastmcp import Client

    instruction = " ".join(sys.argv[1:]) or "a lighthouse on a cliff at sunset"
    print(f"instruction: {instruction!r}\n")

    async with Client(SERVER_DIR / "server.py") as client:  # real stdio server, .env config
        res = await asyncio.wait_for(
            client.call_tool("expand_prompt", {"instruction": instruction}), timeout=1500
        )
        text = next(c for c in res.content if c.type == "text").text
        print(text, "\n")

        m = re.search(r"Recommended size:\s*(\S+)", text)
        if not m:
            print("no recommended size found; skipping generation step")
            return
        ratio = m.group(1).rstrip('"').rstrip(",")
        caption = text.splitlines()[1]

        print(f"[2] generating from the expansion at size={ratio!r} ...")
        res2 = await asyncio.wait_for(
            client.call_tool("generate_image", {"prompt": caption, "size": ratio}), timeout=700
        )
        texts = [c.text for c in res2.content if c.type == "text"]
        images = [c for c in res2.content if c.type == "image"]
        paths = [line.strip() for t in texts for line in t.splitlines() if line.strip().lower().endswith(".png")]
        print(f"    -> {len(images)} inline image(s)")
        for p in paths:
            w, h = png_dims(p)
            print(f"    -> {p}  actual {w}x{h}")
        print("LIVE EXPANSION CHECK COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
