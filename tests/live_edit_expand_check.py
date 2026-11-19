# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Live edit-mode expansion check: expand an edit instruction against a real
image with the official PE edit prompt, then apply the expansion with edit_image.

Usage:  uv run tests/live_edit_expand_check.py [image_path] [instruction]
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


def newest_image() -> str | None:
    gen = SERVER_DIR / "generated"
    if not gen.is_dir():
        return None
    cands = sorted(gen.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else None


async def main() -> None:
    from fastmcp import Client

    img = sys.argv[1] if len(sys.argv) > 1 else newest_image()
    assert img, "no image path given and none found in generated/"
    instruction = sys.argv[2] if len(sys.argv) > 2 else "turn the scene into a foggy early morning"
    print(f"image:       {img}")
    print(f"instruction: {instruction!r}\n")

    async with Client(SERVER_DIR / "server.py") as client:  # real stdio server, .env config
        res = await asyncio.wait_for(
            client.call_tool("expand_prompt", {"instruction": instruction, "image_paths": [img]}),
            timeout=1500,
        )
        text = next(c for c in res.content if c.type == "text").text
        print(text[:1500] + ("..." if len(text) > 1500 else ""), "\n")

        lines = text.splitlines()
        caption = lines[1] if len(lines) > 1 else ""
        m = re.search(r"Target aspect ratio:\s*\"?(\d+:\d+)", text)
        assert caption and m, "expansion output not parseable"
        ratio = m.group(1)

        print(f"[2] applying the expansion with edit_image at size={ratio} ...")
        res2 = await asyncio.wait_for(
            client.call_tool("edit_image", {"prompt": caption, "image_paths": [img], "size": ratio}),
            timeout=700,
        )
        texts = [c.text for c in res2.content if c.type == "text"]
        images = [c for c in res2.content if c.type == "image"]
        paths = [l.strip() for t in texts for l in t.splitlines() if l.strip().lower().endswith(".png")]
        print(f"    -> {len(images)} inline image(s)")
        for p in paths:
            w, h = png_dims(p)
            print(f"    -> {p}  actual {w}x{h}")
        print("LIVE EDIT-EXPANSION CHECK COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
