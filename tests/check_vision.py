# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Check which configured chat models accept image input (needed for edit-PE)."""

import base64
import sys
from pathlib import Path

import httpx

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

import server as srv  # noqa: E402


def main() -> None:
    png = base64.b64encode((SERVER_DIR / "generated-test" / "big-noise.png").read_bytes()) if (SERVER_DIR / "generated-test" / "big-noise.png").is_file() else None
    # 1x1 red PNG fallback
    if not png:
        import struct, zlib
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        raw = b"\x00" + b"\xff\x00\x00"
        png_b = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        png = base64.b64encode(png_b)
    data_url = "data:image/png;base64," + png.decode()

    models = sys.argv[1:] or ["qwen3.8-27b", "glm-5.3-flash"]
    headers = {"Authorization": f"Bearer {srv.API_KEY}"} if srv.API_KEY else {}
    for model in models:
        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly one word: the dominant color of this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens": 20,
        }
        try:
            r = httpx.post(f"{srv.BASE_URL}/chat/completions", headers=headers, json=body, timeout=120)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content", "")
            print(f"{model}: VISION OK  -> answer: {content.strip()[:60]!r}")
        except Exception as exc:
            msg = str(exc)[:200]
            try:
                msg = msg or str(exc.response.text[:200])  # type: ignore[union-attr]
            except Exception:
                pass
            print(f"{model}: NO VISION -> {msg}")


if __name__ == "__main__":
    main()
