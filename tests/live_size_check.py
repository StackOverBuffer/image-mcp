# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Live size check: generates one image at a given size and reports the actual
PNG dimensions, so you can see whether the backend (vLLM behind LiteLLM)
honors the `size` parameter.

Usage:  uv run tests/live_size_check.py "16:9"     (or "2048x2048", "21:9", ...)
"""

import asyncio
import struct
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

PROMPT = "a red apple on a white table, studio lighting, photorealistic"


def png_dims(path: str) -> tuple[int, int]:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", data[16:24])


async def main() -> None:
    from fastmcp import Client

    import server as srv

    size_arg = sys.argv[1] if len(sys.argv) > 1 else "16:9"
    resolved, warnings = srv._resolve_size(size_arg)
    print(f"requested: {size_arg!r} -> sent to API: {resolved}")
    for w_text in warnings:
        print(f"  {w_text}")

    async with Client(srv.mcp) as client:
        res = await asyncio.wait_for(
            client.call_tool("generate_image", {"prompt": PROMPT, "size": size_arg}),
            timeout=620,
        )
    texts = [c.text for c in res.content if c.type == "text"]
    paths = [line.strip() for line in "\n".join(texts).splitlines() if line.strip().lower().endswith(".png")]
    assert paths, texts
    w, h = png_dims(paths[0])
    print(f"saved:     {paths[0]} ({Path(paths[0]).stat().st_size // 1024} KB)")
    print(f"actual:    {w}x{h}")

    if resolved and "x" in resolved:
        rw, rh = (int(v) for v in resolved.split("x"))
        match = (w, h) == (rw, rh)
    else:
        match = True
    print("RESULT:    size HONORED by backend" if match else "RESULT:    size IGNORED by backend (check vLLM/LiteLLM routing)")


if __name__ == "__main__":
    asyncio.run(main())
