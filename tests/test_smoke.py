# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Smoke test for the image MCP server.

Spins up a fake LiteLLM image API (OpenAI-images dialect), then drives the
server's tools twice: in-memory (tool logic) and over real stdio (wiring).
No network or API key needed - the mock only echoes valid responses.
"""

import asyncio
import base64
import json
import os
import re
import struct
import sys
import tempfile
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))


# ------------------------------------------------------------------ fake PNG

def make_png(width: int = 8, height: int = 8, rgb: tuple = (255, 0, 0), raw_rows: bytes | None = None) -> bytes:
    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    if raw_rows is None:
        raw_rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_rows))
        + chunk(b"IEND", b"")
    )


PNG = make_png()
LAST_REQ: dict = {}  # last request seen by the mock, for contract assertions


# ------------------------------------------------------------------ mock API

class MockLiteLLM(BaseHTTPRequestHandler):
    """Fake of LiteLLM's /v1/images/generations and /v1/images/edits."""

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.path == "/v1/images/generations":
            req = json.loads(body)
            LAST_REQ["gen"] = dict(req)
            assert req.get("model") == "minimax-h3-image", "model field missing/wrong"
            assert req.get("prompt"), "prompt field missing"
            if "FAIL" in req["prompt"]:
                self._send(400, {"error": {"message": "content safety check failed (code 1027)"}})
                return
            n = int(req.get("n", 1))
            self._send(200, {"created": 1, "data": [
                {"b64_json": base64.b64encode(PNG).decode()} for _ in range(n)
            ]})
        elif self.path == "/v1/images/edits":
            assert b'name="prompt"' in body, "prompt form field missing"
            assert b'name="model"' in body, "model form field missing"
            n_files = body.count(b'filename=')
            assert n_files >= 1, "no image file parts received"
            m = re.search(rb'name="size"\r\n\r\n([^\r\n]+)', body)
            LAST_REQ["edit"] = {"size": m.group(1).decode() if m else None, "files": n_files}
            self._send(200, {"created": 1, "data": [
                {"b64_json": base64.b64encode(PNG).decode()} for _ in range(n_files)
            ]})
        elif self.path == "/v1/chat/completions":
            req = json.loads(body)
            LAST_REQ["chat"] = req
            blob = json.dumps(req.get("messages", []))
            if "finish_reason:length" in blob:
                self._send(200, {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "{"}}]})
                return
            if "Reference Media" in blob:
                content = json.dumps({"expanded_caption": "EDIT CAPTION", "target_aspect_ratio": "1:1"})
            else:
                content = json.dumps({"expanded_caption": "T2I CAPTION", "aspect_ratio": "16:9"})
            self._send(200, {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def log_message(self, *args) -> None:
        pass  # keep test output clean


def start_mock() -> tuple[str, ThreadingHTTPServer]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockLiteLLM)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


# ------------------------------------------------------------------ tests

async def run_tests() -> None:
    from fastmcp import Client

    base_url, httpd = start_mock()
    out_dir = SERVER_DIR / "generated-test"
    os.environ["LITELLM_BASE_URL"] = base_url
    os.environ["LITELLM_API_KEY"] = "test-key"
    os.environ["IMAGE_OUTPUT_DIR"] = str(out_dir)
    os.environ["PROMPT_LLM_MODEL"] = "qwen3.8-27b"

    import server as srv

    # 0) _default_output_dir: temp-dir script -> stable %TMP% folder; local script -> unchanged
    under_temp = Path(tempfile.gettempdir()) / "somewhere" / "else"
    expected_temp = Path(tempfile.gettempdir()) / "image-mcp" / "generated"
    assert srv._default_output_dir(under_temp) == expected_temp, srv._default_output_dir(under_temp)
    outside_temp = Path(str(Path(tempfile.gettempdir()).resolve()) + "-not-temp") / "repo"
    assert srv._default_output_dir(outside_temp) == outside_temp / "generated"
    print("[PASS] _default_output_dir: temp-dir fallback + unchanged local default", flush=True)

    client = Client(srv.mcp)
    async with client:
        # 1) tool listing
        tools = {t.name for t in await client.list_tools()}
        assert tools == {
            "generate_image", "edit_image", "get_image", "expand_prompt", "get_config", "list_images",
            "scale_image",
        }, tools
        print(f"[PASS] tools listed: {sorted(tools)}", flush=True)

        # 2) generation, n=2
        res = await client.call_tool("generate_image", {"prompt": "a cat", "n": 2})
        images = [c for c in res.content if c.type == "image"]
        text = next(c for c in res.content if c.type == "text").text
        assert len(images) == 2, f"expected 2 images, got {len(images)}"
        paths = [line for line in text.splitlines() if line.strip().endswith(".png")]
        assert len(paths) == 2, text
        for p in paths:
            data = Path(p).read_bytes()
            assert data.startswith(b"\x89PNG"), f"saved file is not a PNG: {p}"
        # localhost image host: markdown link present and actually serves the bytes
        m = re.search(r"http://127\.0\.0\.1:(\d+)/([0-9a-f]+)/([^)\s]+\.png)", text)
        assert m, f"no markdown image link in tool output: {text}"
        assert "include this markdown" in text
        url = f"http://127.0.0.1:{m.group(1)}/{m.group(2)}/{m.group(3)}"
        served = httpx.get(url, timeout=10)
        assert served.status_code == 200, url
        assert served.headers["content-type"].startswith("image/png")
        assert served.content == Path(paths[0]).read_bytes()
        base_url_img = url.rsplit("/", 1)[0]
        traversal = httpx.get(f"{base_url_img}/../../server.py", timeout=10)
        assert traversal.status_code == 404, "path traversal not blocked"
        print(f"[PASS] generate_image n=2 -> {len(images)} inline images, files saved, markdown link serves bytes", flush=True)

        # 3) edit with 2 input files -> mock echoes one image per received file,
        #    proving the multipart contract (image[] fields + prompt + model)
        res2 = await client.call_tool("edit_image", {
            "prompt": "make it blue", "image_paths": paths,
        })
        images2 = [c for c in res2.content if c.type == "image"]
        assert len(images2) == 2, f"edit sent wrong number of files: {len(images2)}"
        text2 = next(c for c in res2.content if c.type == "text").text
        new_paths = [line for line in text2.splitlines() if line.strip().endswith(".png")]
        for p in new_paths:
            assert Path(p).is_file(), f"edited image not saved: {p}"
        print(f"[PASS] edit_image with 2 files -> mock received 2 file parts, {len(images2)} images back", flush=True)

        # 3b) size handling: ratio anchors, exact passthrough, computed ratio,
        #     size on edits, and invalid input rejected with guidance
        await client.call_tool("generate_image", {"prompt": "a cat", "size": "16:9"})
        assert LAST_REQ["gen"]["size"] == "5472x3072", LAST_REQ["gen"]
        await client.call_tool("generate_image", {"prompt": "a cat", "size": "2048x2048"})
        assert LAST_REQ["gen"]["size"] == "2048x2048", LAST_REQ["gen"]
        await client.call_tool("generate_image", {"prompt": "a cat", "size": "3:2"})
        assert LAST_REQ["gen"]["size"] == "5024x3360", LAST_REQ["gen"]
        await client.call_tool("edit_image", {"prompt": "make it blue", "image_paths": paths, "size": "9:16"})
        assert LAST_REQ["edit"]["size"] == "3072x5472", LAST_REQ["edit"]
        try:
            await client.call_tool("generate_image", {"prompt": "a cat", "size": "banana"})
            raise AssertionError("expected invalid size to fail")
        except Exception as exc:
            assert "W:H" in str(exc), f"guidance lost: {exc}"
        await client.call_tool("generate_image", {"prompt": "a cat", "seed": 42})
        assert LAST_REQ["gen"].get("seed") == 42, LAST_REQ["gen"]
        print("[PASS] size: 16:9->5472x3072, WxH passthrough, 3:2->5024x3360, edit 9:16->3072x5472, invalid rejected, seed passthrough", flush=True)

        # 4) expand_prompt: T2I mode -> chat completions with json_schema + official prompts
        res_e = await client.call_tool("expand_prompt", {"instruction": "a lighthouse on a cliff at sunset"})
        e_text = next(c for c in res_e.content if c.type == "text").text
        assert "T2I CAPTION" in e_text and "16:9" in e_text, e_text
        chat = LAST_REQ["chat"]
        assert chat["model"] == "qwen3.8-27b", chat["model"]
        assert chat["messages"][0]["role"] == "system"
        assert "caption-expansion" in chat["messages"][1]["content"]
        assert chat["response_format"]["type"] == "json_schema"
        assert chat["max_tokens"] == 4096, chat["max_tokens"]
        assert chat["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
        print("[PASS] expand_prompt T2I: official system+task prompts, json_schema request", flush=True)

        # 4b) expand_prompt: edit mode with a reference image -> Reference Media parts + adaptive
        res_e2 = await client.call_tool("expand_prompt", {"instruction": "make it night", "image_paths": [paths[0]]})
        e_text2 = next(c for c in res_e2.content if c.type == "text").text
        assert "EDIT CAPTION" in e_text2 and "1:1" in e_text2, e_text2
        blob = json.dumps(LAST_REQ["chat"]["messages"])
        assert "Reference Media: <Picture 1> aspect_ratio 1:1" in blob, blob[:500]
        assert "User Editing Instruction" in blob and "adaptive" in blob
        assert "data:image/jpeg;base64," in blob  # downscaled preview embedded
        assert LAST_REQ["chat"]["max_tokens"] == 16384, LAST_REQ["chat"]["max_tokens"]
        assert "reasoning_effort" not in LAST_REQ["chat"]  # edit params file does not set it
        print("[PASS] expand_prompt edit: Reference Media parts, preview data URL, adaptive", flush=True)

        # 4c) unfinished expansion surfaces a clear error
        try:
            await client.call_tool("expand_prompt", {"instruction": "finish_reason:length please"})
            raise AssertionError("expected unfinished expansion to fail")
        except Exception as exc:
            assert "finish_reason" in str(exc) and "max_tokens was" in str(exc), str(exc)
        print("[PASS] expand_prompt: unfinished generation surfaced clearly", flush=True)

        # 5) error surfacing (provider error must reach the agent verbatim)
        try:
            await client.call_tool("generate_image", {"prompt": "FAIL this request"})
            raise AssertionError("expected an error for FAIL prompt")
        except Exception as exc:
            assert "1027" in str(exc), f"provider error message lost: {exc}"
        print("[PASS] provider error surfaced verbatim (content-safety code visible)", flush=True)

        # 5) get_image re-serves a saved file (with a chat-renderable markdown link)
        res3 = await client.call_tool("get_image", {"path": paths[0]})
        assert any(c.type == "image" for c in res3.content)
        text3 = next(c for c in res3.content if c.type == "text").text
        assert "include this markdown" in text3, text3
        print("[PASS] get_image re-serves saved file + markdown link", flush=True)

        # 5d) list_images: newest first, valid paths
        res_l = await client.call_tool("list_images", {"limit": 3})
        l_text = next(c for c in res_l.content if c.type == "text").text
        l_paths = [line.split("  (")[0].strip() for line in l_text.splitlines() if " MB," in line]
        assert len(l_paths) == 3, l_text
        for lp in l_paths:
            assert Path(lp).is_file(), lp
            assert str(out_dir) in lp, lp
        print("[PASS] list_images: newest-first listing with valid paths", flush=True)

        # 5e) scale_image: factor x2, aspect-preserving width, invalid rejected
        res_s = await client.call_tool("scale_image", {"path": paths[0], "factor": 2})
        s_text = next(c for c in res_s.content if c.type == "text").text
        s_paths = [line for line in s_text.splitlines() if line.strip().lower().endswith(".png")]
        assert len(s_paths) == 1, s_text
        w1, h1 = struct.unpack(">II", Path(s_paths[0]).read_bytes()[16:24])
        assert (w1, h1) == (16, 16), (w1, h1)  # 8x8 source doubled
        sq = out_dir / "scale-src.png"
        sq.write_bytes(make_png(32, 16))
        res_s2 = await client.call_tool("scale_image", {"path": str(sq), "width": 64})
        s2_text = next(c for c in res_s2.content if c.type == "text").text
        s2_paths = [line for line in s2_text.splitlines() if line.strip().lower().endswith(".png")]
        w2, h2 = struct.unpack(">II", Path(s2_paths[0]).read_bytes()[16:24])
        assert (w2, h2) == (64, 32), (w2, h2)  # aspect preserved
        assert "include this markdown" in s2_text
        try:
            await client.call_tool("scale_image", {"path": paths[0]})
            raise AssertionError("expected missing size to fail")
        except Exception as exc:
            assert "factor" in str(exc), str(exc)
        try:
            await client.call_tool("scale_image", {"path": paths[0], "quality": 80, "max_mb": 1.0})
            raise AssertionError("expected quality+max_mb conflict to fail")
        except Exception as exc:
            assert "not both" in str(exc), str(exc)
        # compression: max_mb ladder on a large noise image (png source -> jpg),
        # plus same-size quality recompress
        big_scale = out_dir / "scale-big.png"
        big_scale.write_bytes(make_png(1200, 1200, raw_rows=b"".join(b"\x00" + os.urandom(1200 * 3) for _ in range(1200))))
        res_c = await client.call_tool("scale_image", {"path": str(big_scale), "max_mb": 1.0})
        c_text = next(c for c in res_c.content if c.type == "text").text
        c_paths = [line for line in c_text.splitlines() if line.strip().lower().endswith(".jpg")]
        assert len(c_paths) == 1, c_text
        assert Path(c_paths[0]).stat().st_size <= 1_000_000, c_text
        res_q = await client.call_tool("scale_image", {"path": s_paths[0], "quality": 80, "fmt": "jpg"})
        q_text = next(c for c in res_q.content if c.type == "text").text
        q_paths = [line for line in q_text.splitlines() if line.strip().lower().endswith(".jpg")]
        assert len(q_paths) == 1 and "64x32 (jpg, q80" in q_text, q_text  # same-size recompress, quality applied
        print("[PASS] scale_image: factor x2, aspect width, max_mb compression, quality recompress, conflicts rejected", flush=True)

        # 5c) get_config: effective config visible, key never leaked
        res_c = await client.call_tool("get_config", {})
        c_text = next(c for c in res_c.content if c.type == "text").text
        assert "test-key" not in c_text, "API key leaked via get_config"
        assert "api_key" in c_text and "set (" in c_text, c_text
        assert base_url in c_text and "qwen3.8-27b" in c_text  # prompt model picked up from env
        print("[PASS] get_config: effective env config reported, key masked", flush=True)

        # 5b) inline previews are downscaled (full-res file stays on disk)
        big = out_dir / "big-noise.png"
        noise = b"".join(b"\x00" + os.urandom(1600 * 3) for _ in range(1600))
        big.write_bytes(make_png(1600, 1600, raw_rows=noise))
        res_big = await client.call_tool("get_image", {"path": str(big)})
        img_block = next(c for c in res_big.content if c.type == "image")
        decoded = base64.b64decode(img_block.data)
        assert img_block.mime_type == "image/jpeg", img_block.mime_type
        assert len(decoded) < 1_000_000, f"preview not downscaled: {len(decoded)} bytes"
        assert len(decoded) < big.stat().st_size // 4
        print(f"[PASS] preview downscale: {big.stat().st_size // 1024} KB PNG -> {len(decoded) // 1024} KB jpeg preview", flush=True)

    # 6) real stdio round-trip: spawn server.py as a subprocess and talk MCP to it
    #    (mock stays UP so the stdio tool call hits a live fake API)
    from fastmcp.client.transports import StdioTransport

    print("[....] starting stdio subprocess ...", flush=True)
    env = {
        **os.environ,
        "LITELLM_BASE_URL": base_url,
        "IMAGE_OUTPUT_DIR": str(out_dir),
    }
    transport = StdioTransport(sys.executable, [str(SERVER_DIR / "server.py")], env=env)
    stdio_client = Client(transport)
    try:
        await asyncio.wait_for(stdio_client.__aenter__(), timeout=30)
        print("[PASS] stdio handshake (initialize) OK", flush=True)
        stdio_tools = {t.name for t in await asyncio.wait_for(stdio_client.list_tools(), timeout=30)}
        assert stdio_tools == {
            "generate_image", "edit_image", "get_image", "expand_prompt", "get_config", "list_images",
            "scale_image",
        }, stdio_tools
        print(f"[PASS] stdio tool list: {sorted(stdio_tools)}", flush=True)
        res4 = await asyncio.wait_for(
            stdio_client.call_tool("generate_image", {"prompt": "over stdio"}), timeout=60
        )
        assert any(c.type == "image" for c in res4.content)
        print("[PASS] stdio generate_image returned an image", flush=True)
    finally:
        try:
            await asyncio.wait_for(stdio_client.__aexit__(None, None, None), timeout=15)
        except Exception as exc:
            print(f"[WARN] stdio client close: {exc}", flush=True)
    print("[PASS] stdio round-trip: subprocess start, tool list, tool call", flush=True)

    httpd.shutdown()
    httpd.server_close()

    # keep one artifact for manual inspection
    kept = sorted(out_dir.glob("*.png"))
    print(f"[PASS] {len(kept)} test images kept in {out_dir}", flush=True)
    print("ALL SMOKE TESTS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(run_tests())
