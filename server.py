#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Image generation & editing MCP server.

Talks to a LiteLLM proxy that routes an OpenAI-images-compatible image model
(built for MiniMax-H3-Image, works with anything exposing
/v1/images/generations + /v1/images/edits).

Tools
  generate_image(prompt, size?, n?)   text -> image(s), saved to disk
  edit_image(prompt, image_paths, size?, n?)  image(s) + instruction -> edited image(s)
  get_image(path)                     re-serve a previously saved image

Every generation/edit tool returns the saved absolute path(s) as text AND the
image(s) themselves as image content, so a vision-capable agent can review
the result immediately and iterate with edit_image until it matches.

Configuration (environment variables, or a .env file next to this script):
  LITELLM_BASE_URL   e.g. http://localhost:4000           (required)
  LITELLM_API_KEY    bearer token, if your proxy needs one
  IMAGE_MODEL        model name, default: minimax-h3-image
  IMAGE_OUTPUT_DIR   where to save images; default: <script dir>/generated, or
                     <system temp dir>/image-mcp/generated when the script runs
                     from a temp dir itself (e.g. `uv run <URL>`)
  IMAGE_TIMEOUT      request timeout in seconds, default: 600
  IMAGE_PREVIEW_MAX  max side (px) of the inline preview the agent sees, default: 1024;
                     0 = inline full-res. Full-res files are always saved to disk.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from pathlib import Path

import httpx
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

try:
    from fastmcp.exceptions import ToolError
except ImportError:  # older fastmcp
    from fastmcp.exceptions import ResourceError as ToolError  # type: ignore

from mcp.types import TextContent

# ---------------------------------------------------------------- config

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"


def _load_dotenv() -> None:
    """Tiny .env loader (real environment variables always win)."""
    env_file = SCRIPT_DIR / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

BASE_URL = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("IMAGE_MODEL", "minimax-h3-image")
def _default_output_dir(script_dir: Path) -> Path:
    """Output dir used when IMAGE_OUTPUT_DIR is unset.

    Normally <script_dir>/generated. When the script itself runs from a temp
    location (e.g. `uv run <URL>` executes the fetched script from uv's temp
    dir), fall back to a stable folder under the system temp dir - a per-run
    random dir would break get_image/list_images across restarts.
    """
    script_dir = script_dir.resolve()
    temp = Path(tempfile.gettempdir()).resolve()
    if script_dir.is_relative_to(temp):
        return temp / "image-mcp" / "generated"
    return script_dir / "generated"


OUTPUT_DIR = Path(
    os.environ.get("IMAGE_OUTPUT_DIR") or _default_output_dir(SCRIPT_DIR)
).expanduser()
TIMEOUT = float(os.environ.get("IMAGE_TIMEOUT", "600"))
PREVIEW_MAX = int(os.environ.get("IMAGE_PREVIEW_MAX", "1024"))
PROMPT_LLM_MODEL = os.environ.get("PROMPT_LLM_MODEL", "")
PROMPT_VLM_MODEL = os.environ.get("PROMPT_VLM_MODEL", "")  # falls back to PROMPT_LLM_MODEL
PROMPT_TIMEOUT = float(os.environ.get("PROMPT_TIMEOUT", "1200"))
IMAGE_HTTP_PORT = int(os.environ.get("IMAGE_HTTP_PORT", "18090"))  # -1 disables, 0 = random port

if not BASE_URL.endswith("/v1"):
    BASE_URL += "/v1"

MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

mcp = FastMCP(
    "minimax-image",
    instructions=(
        "Image generation & editing. generate_image creates images from text. "
        "edit_image changes existing image(s) - reference them in the prompt as "
        "'image 1', 'image 2', ... in the order given. expand_prompt turns a "
        "short idea or edit instruction into a detailed caption plus an "
        "aspect-ratio recommendation, using MiniMax's official prompt-expansion "
        "prompts. Use expand_prompt by default for generation - even when you "
        "could write a detailed prompt yourself, its captions match the style "
        "the image model was trained on; skip it only when the user already "
        "supplied a complete caption. For edits, use it when the instruction "
        "doesn't say where/what exactly should change (an unspecified change "
        "risks a literal misread) and skip it when the edit is fully "
        "specified. Follow its recommended aspect ratio when the user didn't "
        "pick one; for edits you can pass its caption verbatim (<Picture "
        "1>/<Subject 1> placeholders are the official format) or tighten it "
        "first - the expanded caption is an editable spec. Every result comes "
        "back as an inline image you can see: review "
        "it. If it is fundamentally off (wrong composition or subject), "
        "regenerate with a sharpened prompt instead of editing a bad base; if "
        "only details are off, call edit_image with a targeted instruction. If "
        "two attempts share the same flaw, rewrite the prompt instead of rolling "
        "again. Iterate until it matches."
    ),
)


# ---------------------------------------------------------------- helpers


def _require_config() -> None:
    if not BASE_URL:
        raise ToolError(
            "LITELLM_BASE_URL is not configured. Set it via environment variables "
            "(e.g. in the MCP client config's env block) or in the .env file next to "
            "server.py, e.g. LITELLM_BASE_URL=https://litellm.example.com"
        )


def _headers() -> dict[str, str]:
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _save_images(images: list[tuple[bytes, str]], prefix: str = "IMG") -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _start_image_server()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = []
    for i, (data, ext) in enumerate(images, 1):
        suffix = f"-{i}" if len(images) > 1 else ""
        path = OUTPUT_DIR / f"{prefix}-{stamp}{suffix}.{ext}"
        path.write_bytes(data)
        paths.append(path.resolve())
    return paths


def _fetch_image_url(url: str) -> tuple[bytes, str]:
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    ext = ".png"
    ctype = resp.headers.get("content-type", "")
    for known_ext, mime in MIME_BY_EXT.items():
        if mime in ctype:
            ext = known_ext
            break
    return resp.content, ext.lstrip(".")


def _extract_images(payload: dict) -> list[tuple[bytes, str]]:
    """Pull images out of an OpenAI images API response (b64_json or url)."""
    images = []
    for item in payload.get("data") or []:
        if item.get("b64_json"):
            images.append((base64.b64decode(item["b64_json"]), "png"))
        elif item.get("url"):
            images.append(_fetch_image_url(item["url"]))
    if not images:
        raise ToolError(f"Model returned no images. Response: {str(payload)[:500]}")
    return images


def _preview_jpeg_bytes(path: Path) -> bytes:
    """Downscaled JPEG copy for inline contexts (agent vision, VLM input)."""
    if PREVIEW_MAX <= 0:
        return path.read_bytes()
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _preview_block(path: Path):
    """Downscaled inline copy for the agent's vision context (file on disk stays full-res)."""
    if PREVIEW_MAX <= 0:
        return Image(data=path.read_bytes(), format=path.suffix.lstrip(".")).to_image_content()
    return Image(data=_preview_jpeg_bytes(path), format="jpeg").to_image_content()


def _result_blocks(paths: list[Path], note: str) -> list:
    md_links = [f"![{p.name}]({link})" for p in paths if (link := _markdown_link(p))]
    if md_links:
        note += (
            "\n\nTo show the image(s) in the chat window, include this markdown in your reply:\n"
            + "\n".join(md_links)
        )
    blocks: list = [TextContent(type="text", text=note + "\n" + "\n".join(str(p) for p in paths))]
    for path in paths:
        blocks.append(_preview_block(path))
    return blocks


# ------------------------------------------------- localhost image host

_HTTP_LOCK = threading.Lock()
_HTTP_STATE: dict = {"server": None, "port": 0, "token": ""}


class _ImageHTTPServer(ThreadingHTTPServer):
    # HTTPServer enables SO_REUSEADDR; on Windows that silently allows co-binding
    # a port another instance is already listening on (connections keep going to
    # the first binder). Disable it so a busy port fails the bind and we fall
    # back to a random one - every instance's links then carry its own port.
    allow_reuse_address = False
    daemon_threads = True


def _start_image_server() -> None:
    """Read-only HTTP host for OUTPUT_DIR so chat clients can render images by URL.

    Binds 127.0.0.1 only, serves files under OUTPUT_DIR behind a per-session random
    token path, sends Cache-Control: no-store so updated files always display.
    """
    with _HTTP_LOCK:
        if IMAGE_HTTP_PORT < 0 or _HTTP_STATE["server"] is not None:
            return
        token = os.urandom(8).hex()
        output_root = OUTPUT_DIR.resolve()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                prefix = f"/{token}/"
                if not path.startswith(prefix):
                    self.send_error(404)
                    return
                candidate = (OUTPUT_DIR / path[len(prefix):]).resolve()
                try:
                    candidate.relative_to(output_root)
                except ValueError:
                    self.send_error(404)
                    return
                if not candidate.is_file():
                    self.send_error(404)
                    return
                data = candidate.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", MIME_BY_EXT.get(candidate.suffix.lower(), "application/octet-stream"))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass

        port = IMAGE_HTTP_PORT
        server = None
        for _ in range(2):  # preferred port first, then a random one
            try:
                server = _ImageHTTPServer(("127.0.0.1", port), Handler)
                break
            except OSError:
                port = 0
        if server is None:
            return
        threading.Thread(target=server.serve_forever, daemon=True, name="image-http").start()
        _HTTP_STATE.update(server=server, port=server.server_address[1], token=token)


def _markdown_link(path: Path) -> str | None:
    """Chat-renderable URL for a saved image, or None if the image host is off."""
    if IMAGE_HTTP_PORT < 0 or _HTTP_STATE["server"] is None:
        return None
    return f"http://127.0.0.1:{_HTTP_STATE['port']}/{_HTTP_STATE['token']}/{path.name}"


NATIVE_MP = 16.8  # native output target: 4096x4096 (1:1), 5472x3072 (16:9), 6272x2688 (21:9)
SWEET_MP = (4.0, 20.0)  # documented working range


def _resolve_size(size: str | None) -> tuple[str | None, list[str]]:
    """Normalize the size parameter to a "WxH" pixel string.

    Accepts "W:H" aspect ratios (mapped to ~16 MP native resolution, snapped to
    multiples of 32 - reproduces the native anchors exactly) and "WxH" exact
    pixels (passed through). Out-of-envelope values are passed through with a
    warning instead of being refused - backends vary in strictness.
    """
    if size is None or not size.strip():
        return None, []
    s = size.strip().lower().replace("×", "x")
    if "x" in s:
        try:
            w, h = (int(p) for p in s.split("x", 1))
        except ValueError:
            raise ToolError(
                f"Invalid size {size!r}. Use 'W:H' aspect ratio (e.g. '1:1', '16:9', "
                f"'21:9'; range 1:3..3:1) or 'WxH' exact pixels (e.g. '2048x2048')."
            ) from None
    elif ":" in s:
        try:
            wr, hr = (int(p) for p in s.split(":", 1))
        except ValueError:
            raise ToolError(
                f"Invalid size {size!r}. Use 'W:H' aspect ratio (e.g. '1:1', '16:9', "
                f"'21:9'; range 1:3..3:1) or 'WxH' exact pixels (e.g. '2048x2048')."
            ) from None
        if wr <= 0 or hr <= 0:
            raise ToolError(f"Invalid size {size!r}: ratio parts must be positive integers.")
        r = wr / hr
        w = round((NATIVE_MP * 1e6 * r) ** 0.5 / 32) * 32
        h = round((NATIVE_MP * 1e6 / r) ** 0.5 / 32) * 32
    else:
        raise ToolError(
            f"Invalid size {size!r}. Use 'W:H' aspect ratio (e.g. '1:1', '16:9', "
            f"'21:9'; range 1:3..3:1) or 'WxH' exact pixels (e.g. '2048x2048')."
        )
    if w < 64 or h < 64:
        raise ToolError(f"Invalid size {w}x{h}: sides must be at least 64px.")
    warnings: list[str] = []
    ratio = w / h
    if not 1 / 3 <= ratio <= 3:
        warnings.append(
            f"note: aspect {ratio:.2f}:1 is outside the documented 1:3..3:1 range; the backend may reject it"
        )
    mp = w * h / 1e6
    if not SWEET_MP[0] <= mp <= SWEET_MP[1]:
        warnings.append(
            f"note: {mp:.1f} MP is outside the ~4-20 MP sweet spot (native ~16 MP); the backend may adjust or reject it"
        )
    if w % 32 or h % 32:
        warnings.append("note: dimensions are not multiples of 32; the backend may round or reject them")
    return f"{w}x{h}", warnings


def _note(base: str, size: str | None, warnings: list[str]) -> str:
    head = f"{base} at {size}" if size else base
    return "\n".join([head + ":"] + warnings)


def _read_image_file(path_str: str) -> tuple[str, bytes, str]:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise ToolError(f"Image file not found: {path}")
    ext = path.suffix.lower()
    if ext not in MIME_BY_EXT:
        raise ToolError(f"Unsupported image type '{ext}' (use png/jpg/jpeg/webp): {path}")
    return path.name, path.read_bytes(), MIME_BY_EXT[ext]


def _post(endpoint: str, *, json: dict | None = None, files=None, data: dict | None = None, timeout: float | None = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    try:
        with httpx.Client(timeout=timeout or TIMEOUT) as client:
            resp = client.post(url, headers=_headers(), json=json, files=files, data=data)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise ToolError(
            f"Image API returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ToolError(
            f"Could not reach the image API at {url}: {exc}. "
            f"(Generation can take minutes; IMAGE_TIMEOUT is {TIMEOUT:.0f}s.)"
        ) from exc


# ------------------------------------------------- prompt expansion (PE)

T2I_RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16"]


def _prompt_file(*parts: str) -> str:
    path = PROMPTS_DIR.joinpath(*parts)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    text = EMBEDDED_PROMPTS.get("/".join(parts))
    if text is not None:
        return text
    raise ToolError(f"Prompt-expansion file missing: {path}")


def _exact_ratio(w: int, h: int) -> str:
    g = gcd(w, h) or 1
    return f"{w // g}:{h // g}"


def _image_dims(path: Path) -> tuple[int, int]:
    with PILImage.open(path) as im:
        return im.size


def _require_prompt_model(vlm: bool = False) -> str:
    model = PROMPT_VLM_MODEL if vlm and PROMPT_VLM_MODEL else PROMPT_LLM_MODEL
    if model:
        return model
    available: list[str] = []
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(f"{BASE_URL}/models", headers=_headers())
            available = [m.get("id", "") for m in r.json().get("data", []) if "image" not in m.get("id", "").lower()]
    except Exception:
        pass
    hint = f" Chat models on your proxy right now: {', '.join(available)}." if available else ""
    raise ToolError(
        "expand_prompt needs PROMPT_LLM_MODEL (a chat/vision model served by the same "
        "LiteLLM proxy). Set PROMPT_LLM_MODEL via environment variables (e.g. the MCP "
        "client config's env block) or in .env, e.g. PROMPT_LLM_MODEL=glm-5.3-flash." + hint
    )


def _chat_json(body: dict, keys: set[str], max_tokens_used: int | None = None) -> tuple[dict, str | None]:
    note: str | None = None
    try:
        resp = _post("chat/completions", json=body, timeout=PROMPT_TIMEOUT)
    except ToolError as exc:
        msg = str(exc)
        if body.get("response_format") and (
            "response_format" in msg or "json_schema" in msg or "json schema" in msg.lower()
        ):
            fallback = {k: v for k, v in body.items() if k != "response_format"}
            resp = _post("chat/completions", json=fallback, timeout=PROMPT_TIMEOUT)
            note = "note: backend does not support response_format; JSON is enforced by the prompt only."
        else:
            raise
    choices = resp.get("choices") or []
    if not choices:
        raise ToolError(f"Chat model returned no choices: {str(resp)[:300]}")
    choice = choices[0]
    finish = choice.get("finish_reason")
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    if finish not in (None, "stop"):
        budget = f" (max_tokens was {max_tokens_used})" if max_tokens_used else ""
        raise ToolError(
            f"Prompt expansion did not finish (finish_reason={finish!r}){budget}. "
            "Reasoning models can spend most of the completion budget on hidden "
            "thinking before any output - raise max_tokens in the mode's "
            "request_params.json under prompts/."
        )
    if text.startswith("```"):
        text = text.strip("`").lstrip()
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ToolError(f"Chat model returned non-JSON content: {text[:300]}") from None
        try:
            result = json.loads(m.group(0))
        except json.JSONDecodeError:
            raise ToolError(f"Chat model returned unparseable JSON: {text[:300]}") from None
    if not isinstance(result, dict) or not keys.issubset(result):
        raise ToolError(f"Chat model returned unexpected JSON: {str(result)[:300]}")
    return result, note


def _load_params(mode: str) -> dict:
    """Optional per-mode sampling params (e.g. prompts/edit/request_params.json)."""
    path = PROMPTS_DIR / mode / "request_params.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    embedded = EMBEDDED_PROMPTS.get(f"{mode}/request_params.json")
    return json.loads(embedded) if embedded else {}


def _expand_t2i(instruction: str) -> str:
    model = _require_prompt_model()
    params = _load_params("t2i")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _prompt_file("t2i", "system_prompt.txt").strip()},
            {"role": "user", "content": _prompt_file("t2i", "task_prompt.txt").replace("{}", instruction)},
        ],
        "temperature": params.get("temperature", 1.0),
        "top_p": params.get("top_p", 0.95),
        "top_k": params.get("top_k", 20),
        "max_tokens": params.get("max_tokens", 4096),
        "response_format": params.get("response_format"),
        "stream": False,
    }
    for extra_key in ("reasoning_effort", "chat_template_kwargs"):
        if extra_key in params:
            body[extra_key] = params[extra_key]
    result, note = _chat_json(body, {"expanded_caption", "aspect_ratio"}, body.get("max_tokens"))
    ratio = str(result["aspect_ratio"]).strip()
    if ratio not in T2I_RATIOS:
        raise ToolError(f"Expansion returned unsupported aspect ratio {ratio!r} (allowed: {', '.join(T2I_RATIOS)}).")
    caption = str(result["expanded_caption"]).strip()
    if not caption:
        raise ToolError("Expansion returned an empty caption.")
    lines = [
        "Expanded prompt (pass to generate_image):",
        caption,
        f"Recommended size: {ratio}  ->  call generate_image(..., size=\"{ratio}\")",
    ]
    if note:
        lines.append(note)
    return "\n".join(lines)


def _expand_edit(instruction: str, image_paths: list[str], aspect_ratio: str | None) -> str:
    model = _require_prompt_model(vlm=True)
    parts: list[dict] = []
    for i, path_str in enumerate(image_paths, 1):
        path = Path(path_str).expanduser().resolve()
        _read_image_file(path_str)  # validates existence and type
        w, h = _image_dims(path)
        parts.append({"type": "text", "text": f"Reference Media: <Picture {i}> aspect_ratio {_exact_ratio(w, h)}"})
        b64 = base64.b64encode(_preview_jpeg_bytes(path)).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    selection = (aspect_ratio or "adaptive").strip()
    parts.append({
        "type": "text",
        "text": (
            f"User Editing Instruction: {json.dumps(instruction, ensure_ascii=False)}\n"
            f"User Aspect Ratio Selection: {selection}"
        ),
    })
    params = _load_params("edit")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _prompt_file("edit", "task_prompt.txt")},
            {"role": "user", "content": parts},
        ],
        "temperature": params.get("temperature", 1.0),
        "top_p": params.get("top_p", 0.95),
        "top_k": params.get("top_k", 20),
        "max_tokens": params.get("max_tokens", 16384),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "expanded_caption_and_target_ratio",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "expanded_caption": {"type": "string"},
                        "target_aspect_ratio": {"type": "string"},
                    },
                    "required": ["expanded_caption", "target_aspect_ratio"],
                    "additionalProperties": False,
                },
            },
        },
        "stream": False,
    }
    for extra_key in ("reasoning_effort", "chat_template_kwargs"):
        if extra_key in params:
            body[extra_key] = params[extra_key]
    result, note = _chat_json(body, {"expanded_caption", "target_aspect_ratio"}, body.get("max_tokens"))
    ratio = str(result["target_aspect_ratio"]).strip()
    if not re.fullmatch(r"[1-9]\d*:[1-9]\d*", ratio):
        raise ToolError(f"Expansion returned invalid target_aspect_ratio {ratio!r} (expected 'W:H').")
    caption = str(result["expanded_caption"]).strip()
    if not caption:
        raise ToolError("Expansion returned an empty caption.")
    lines = [
        "Expanded edit instruction (pass to edit_image):",
        caption,
        f"Target aspect ratio: {ratio}  ->  call edit_image(..., size=\"{ratio}\")",
    ]
    if note:
        lines.append(note)
    return "\n".join(lines)


# ---------------------------------------------------------------- tools


@mcp.tool
def generate_image(prompt: str, size: str | None = None, n: int = 1, seed: int | None = None) -> list:
    """Generate image(s) from a text description.

    Args:
        prompt: What the image should show. Default to expand_prompt for this:
            hand it the user's idea and use its caption (tightened as needed) -
            the caption style matches what the model was trained on. Write it
            yourself only when the user already supplied a complete caption.
        size: Optional output size. Two forms:
            "W:H" aspect ratio (recommended): mapped to ~16 MP native resolution -
            "1:1" -> 4096x4096, "16:9" -> 5472x3072, "21:9" -> 6272x2688;
            any ratio from 1:3 to 3:1 works.
            "WxH" exact pixels, e.g. "2048x2048" (multiples of 32 are safest).
            Sweet spot ~4-20 MP. Omit for the backend default.
        n: Number of images to generate (default 1). Tip: unsure about composition
            or style? Generate 2 candidates, compare them inline, and keep the
            better one as the base for edit_image.
        seed: Optional integer seed. The same prompt + size + seed reproduces the
            same image (verified on this backend): hold a composition steady while
            tweaking the prompt, or change the seed for an honest re-roll.

    Returns the saved file path(s) and the image(s) themselves for immediate review.
    If the result is fundamentally off (wrong composition, wrong subject, or the
    request was misunderstood), regenerate with a sharpened prompt instead of
    editing a bad base (n=2 candidates help when uncertain). If only details are
    off, refine with edit_image. If two attempts share the same flaw, rewrite the
    prompt - the prompt is the problem, not the model.
    """
    _require_config()
    resolved_size, warnings = _resolve_size(size)
    payload: dict = {"model": MODEL, "prompt": prompt}
    if resolved_size:
        payload["size"] = resolved_size
    if n != 1:
        payload["n"] = n
    if seed is not None:
        payload["seed"] = seed
    payload_json = _post("images/generations", json=payload)
    paths = _save_images(_extract_images(payload_json))
    note = _note(f"Generated {len(paths)} image(s) with {MODEL}", resolved_size, warnings)
    return _result_blocks(paths, note)


@mcp.tool
def edit_image(prompt: str, image_paths: list[str], size: str | None = None, n: int = 1) -> list:
    """Edit existing image(s) with an instruction, or combine them into a new image.

    Args:
        prompt: The change to make. Reference inputs by position: "image 1",
            "image 2", ... matching the order of image_paths
            (e.g. "Replace the background of image 1 with a beach at sunset").
            If the request doesn't pin down where/what exactly should change,
            expand it into a full target description with expand_prompt first
            (unspecified edits risk a literal misread).
        image_paths: Absolute path(s) of image file(s) to edit, 1-16 images.
            Use the paths returned by generate_image/edit_image/get_image.
        size: Optional output size, same formats as generate_image
            ("16:9" or "2048x2048"). Omit to let the backend match the input.
        n: Number of output images (default 1).

    Returns the saved file path(s) and the edited image(s) for immediate review.
    Editing anchors to the input image(s): use it when the base is right and only
    details need to change. If the composition or subject itself is wrong, go back
    to generate_image with a sharper prompt instead of editing a bad base.
    """
    _require_config()
    if not 1 <= len(image_paths) <= 16:
        raise ToolError("image_paths must contain between 1 and 16 image paths.")
    resolved_size, warnings = _resolve_size(size)
    files = []
    for path_str in image_paths:
        name, data, mime = _read_image_file(path_str)
        field = "image" if len(image_paths) == 1 else "image[]"
        files.append((field, (name, data, mime)))
    form: dict = {"model": MODEL, "prompt": prompt}
    if resolved_size:
        form["size"] = resolved_size
    if n != 1:
        form["n"] = str(n)
    payload_json = _post("images/edits", files=files, data=form)
    paths = _save_images(_extract_images(payload_json))
    note = _note(f"Edited {len(image_paths)} input image(s) with {MODEL}", resolved_size, warnings)
    return _result_blocks(paths, note)


@mcp.tool
def scale_image(
    path: str,
    width: int | None = None,
    height: int | None = None,
    factor: float | None = None,
    fmt: str | None = None,
    quality: int | None = None,
    max_mb: float | None = None,
) -> list:
    """Resize an existing image up or down with Lanczos resampling - exact pixels,
    instant, and no generation cost. Also recompresses to shrink file size.

    Args:
        path: Absolute path of the source image.
        width: Target width in px; height keeps the aspect ratio unless height is also given.
        height: Target height in px; width keeps the aspect ratio unless width is also given.
        factor: Scale factor instead of explicit size (0.5 = half, 2.0 = double).
        fmt: Optional output format "png", "jpg" or "webp" (jpg flattens transparency
            onto white).
        quality: Optional JPEG/WebP quality 1-100 (default 95 when encoding lossy).
        max_mb: Optional target file size in MB: picks the highest JPEG/WebP quality
            that lands under it. Without an explicit fmt, a PNG source is re-encoded
            as jpg (PNG cannot hit size targets predictably). Give quality or
            max_mb, not both.

    Give width and/or height, or factor, or compression (quality/max_mb) alone -
    compression without size arguments re-encodes at the current dimensions.
    Returns the saved path (with final size in MB), an inline preview and a chat
    markdown link. Note: upscaling adds pixels, not detail - for a detail-true
    enlargement of a generated image, prefer edit_image(path, size="WxH"), which
    re-renders the scene at the target resolution.
    """
    src = Path(path).expanduser().resolve()
    _read_image_file(path)  # validates existence and type
    with PILImage.open(src) as im:
        w, h = im.size
    if quality is not None and max_mb is not None:
        raise ToolError("Use quality or max_mb, not both.")
    if quality is not None and not 1 <= quality <= 100:
        raise ToolError("quality must be 1-100.")
    ext = (fmt or src.suffix.lstrip(".")).lower().strip(".")
    ext = "jpg" if ext == "jpeg" else ext
    if ext not in ("png", "jpg", "webp"):
        raise ToolError(f"Unsupported output format '{fmt}' (use png/jpg/webp).")
    warnings: list[str] = []
    if max_mb is not None and ext == "png":
        if fmt is not None:
            raise ToolError("max_mb needs a lossy format - set fmt='jpg' or 'webp'.")
        ext = "jpg"  # PNG cannot hit size targets predictably; default to jpg
    if factor is not None:
        if width is not None or height is not None:
            raise ToolError("Use either factor alone, or width and/or height.")
        if not 0 < factor <= 16:
            raise ToolError("factor must be within 0 < factor <= 16.")
        tw, th = max(1, round(w * factor)), max(1, round(h * factor))
    elif width is not None or height is not None:
        if (width is not None and width <= 0) or (height is not None and height <= 0):
            raise ToolError("width/height must be positive.")
        tw = width if width is not None else round(w * height / h)
        th = height if height is not None else round(h * width / w)
    elif quality is not None or max_mb is not None:
        tw, th = w, h  # recompress without resizing
    else:
        raise ToolError("Give width and/or height, or factor, or quality/max_mb to recompress.")
    tw, th = int(tw), int(th)
    if not (8 <= tw <= 8192 and 8 <= th <= 8192):
        raise ToolError(f"Target size {tw}x{th} out of range (8..8192 per side).")
    if quality is not None and ext == "png":
        warnings.append("note: quality applies only to jpg/webp; it is ignored for png output")

    def encode(img, q: int | None) -> bytes:
        buf = io.BytesIO()
        if ext == "png":
            img.save(buf, format="PNG")
        elif ext == "jpg":
            img.save(buf, format="JPEG", quality=95 if q is None else q)
        else:
            img.save(buf, format="WEBP", quality=95 if q is None else q)
        return buf.getvalue()

    with PILImage.open(src) as im:
        if ext == "jpg" and im.mode != "RGB":
            im = im.convert("RGB")
        resized = im.resize((tw, th), PILImage.Resampling.LANCZOS) if (tw, th) != (w, h) else im
        if max_mb is not None:
            target = max_mb * 1e6
            data, used_q = None, None
            for q in (95, 90, 85, 75, 65, 55, 45, 35, 25):
                cand = encode(resized, q)
                if len(cand) <= target:
                    data, used_q = cand, q
                    break
            if data is None:
                data, used_q = encode(resized, 20), 20
                warnings.append(
                    f"note: still {len(data) / 1e6:.1f} MB above the {max_mb} MB target even at quality 20"
                )
        else:
            data, used_q = encode(resized, quality), quality
    out_path = _save_images([(data, ext)], prefix="SCALE")[0]
    q_txt = f", q{used_q}" if used_q is not None and ext != "png" else ""
    note = _note(
        f"Scaled {src.name} ({w}x{h}) to {tw}x{th} ({ext}{q_txt}, {len(data) / 1e6:.1f} MB)",
        None,
        warnings,
    )
    return _result_blocks([out_path], note)


@mcp.tool
def get_image(path: str) -> list:
    """Re-serve a previously saved image so you can look at it again.

    Args:
        path: Absolute path of the image (as returned by the other tools).
    """
    _start_image_server()
    resolved = Path(path).expanduser().resolve()
    _read_image_file(path)  # validates existence and type
    text = str(resolved)
    link = _markdown_link(resolved)
    if link:
        text += (
            "\n\nTo show it in the chat window, include this markdown in your reply:\n"
            f"![{resolved.name}]({link})"
        )
    return [TextContent(type="text", text=text), _preview_block(resolved)]


@mcp.tool
def list_images(limit: int = 10) -> str:
    """List recently saved images (newest first) with path, size and timestamp.

    Use this to find earlier results after context compaction, or to pick a
    base image for edit_image. Pass a listed path to get_image to view it again.
    """
    limit = max(1, min(int(limit), 50))
    if not OUTPUT_DIR.is_dir():
        return f"No images yet (output dir does not exist): {OUTPUT_DIR}"
    files = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in set(MIME_BY_EXT)]
    if not files:
        return f"No images yet in {OUTPUT_DIR}"
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    lines = []
    for p in files[:limit]:
        stat = p.stat()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        lines.append(f"{p}  ({stat.st_size / 1e6:.1f} MB, {stamp})")
    out = "\n".join(lines)
    hidden = len(files) - len(lines)
    if hidden > 0:
        out += f"\n(+{hidden} older files not shown; raise the limit to see more)"
    return out


@mcp.tool
def expand_prompt(instruction: str, image_paths: list[str] | None = None, aspect_ratio: str | None = None) -> str:
    """Turn a short image idea or edit instruction into a detailed caption plus an
    aspect-ratio recommendation, using MiniMax's official H3 prompt-expansion prompts.

    Default to this for generation - even a detailed idea benefits from being
    rewritten in the caption style the image model was trained on; skip it only
    when the user already supplied a complete caption. For edits, use it when
    the instruction doesn't say where/what exactly should change (unspecified
    changes risk a literal misread) and skip it when the edit is fully
    specified. Follow its recommended aspect ratio when the user didn't pick one.

    Args:
        instruction: Short request in any language, e.g. "a lighthouse on a cliff at sunset".
        image_paths: Optional reference image path(s) (1-16). When given, the instruction is
            expanded as an EDIT: the result is a complete target description for edit_image
            (requires a vision-capable PROMPT_LLM_MODEL). When omitted, it is expanded as a
            text-to-image prompt for generate_image.
        aspect_ratio: Edit mode only: pin the target canvas ("16:9", "3:2", ...) or leave
            "adaptive" (default) to let the model choose from the references. In T2I mode
            the expansion recommends a ratio itself.

    Returns the expanded caption and the aspect ratio to pass as size= to
    generate_image or edit_image. Review the expansion and adjust it in your own
    words if needed before generating.
    """
    _require_config()
    if image_paths:
        if not 1 <= len(image_paths) <= 16:
            raise ToolError("image_paths must contain between 1 and 16 image paths.")
        if aspect_ratio and aspect_ratio.strip() != "adaptive" and not re.fullmatch(r"[1-9]\d*:[1-9]\d*", aspect_ratio.strip()):
            raise ToolError("aspect_ratio must be 'adaptive' or a 'W:H' ratio like '16:9' (edit mode).")
        return _expand_edit(instruction, image_paths, aspect_ratio)
    if aspect_ratio:
        raise ToolError(
            "aspect_ratio applies only in edit mode (when image_paths is given); "
            "T2I expansion recommends a ratio itself."
        )
    return _expand_t2i(instruction)


@mcp.tool
def get_config() -> str:
    """Show the server's active configuration. The API key is never included - only whether it is set.

    Useful to verify that environment variables (from the MCP client config or a
    .env file) were picked up: ask for the config and compare against expectations.
    """
    return "\n".join([
        f"base_url:       {BASE_URL or '(not configured)'}",
        f"api_key:        {'set (' + str(len(API_KEY)) + ' chars)' if API_KEY else 'not set'}",
        f"image_model:    {MODEL}",
        f"output_dir:     {OUTPUT_DIR}",
        f"image_timeout:  {TIMEOUT:.0f}s",
        f"preview_max:    {PREVIEW_MAX}px (0 = inline full-res)",
        f"prompt_llm:     {PROMPT_LLM_MODEL or '(not configured)'}",
        f"prompt_vlm:     {PROMPT_VLM_MODEL or '(same as prompt_llm)'}",
        f"prompt_timeout: {PROMPT_TIMEOUT:.0f}s",
        f"pe_prompts_dir: {PROMPTS_DIR}"
        + ("" if PROMPTS_DIR.is_dir() else " (missing - using embedded prompts)"),
    ])


# ------------------------------------------------- embedded PE prompt assets
# Vendored MiniMax-H3-Image-PE prompts. Source of truth: prompts/ (tracked by
# manifest.json). Files on disk win when present (repo installs stay
# live-tweakable); this embedded copy is the fallback for single-file
# deployments (e.g. `uv run <URL>`) where prompts/ does not exist.
# BEGIN EMBEDDED PROMPTS (generated block - do not edit)
EMBEDDED_PROMPTS: dict[str, str] = {
    't2i/system_prompt.txt': 'Reason in the thinking section first. The visible final content must be exactly one JSON object with exactly two keys: expanded_caption and aspect_ratio. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \\". Do not output text outside that object.\n',
    't2i/task_prompt.txt': 'You are a caption-expansion and canvas-selection model for still-image\ntext-to-image generation. Rewrite input_text into a natural English detailed\ncaption and choose one aspect ratio for the requested final image. After private\nreasoning, serialize the complete caption and aspect ratio together as the one\nrequired two-key JSON object; no visible analysis, markdown, or self-talk. The\ncaption may be long when the requested image is a complex information or design artifact.\n\nRole: act like an expert visual designer and technical illustrator, not a word\nexpander. Preserve every hard fact from the input, then complete the image with\nvisible, inspectable content. The goal is not more words; the goal is that a\nviewer can inspect each important part of the image and find meaningful subjects,\nlabels, relationships, materials, data, and hierarchy.\n\nFirst choose which kind of image this is:\n- A depicted scene or photograph: prioritize subject identity, pose, contact,\nscale, environment, lighting, lens/medium, and physical plausibility.\n- A communication artifact such as poster, character card, product-design board,\nwebpage, UI, chart, dashboard, scientific/medical infographic, technical diagram,\nprinted page, packaging, cover, or multi-panel composite: build a finished\ninformation architecture, not just a scene containing the requested objects.\n\nAspect-ratio decision policy:\nChoose exactly one of these allowed ratios: 1:1, 5:4, 4:3, 3:2, 16:9, 21:9,\n4:5, 3:4, 2:3, or 9:16. Decide in the following order; earlier evidence\noverrides later evidence.\n\n1. Explicit final canvas\n- Preserve an explicit final-output ratio. Map explicit output pixel dimensions\n  by width divided by height to the closest allowed ratio.\n- Do not confuse the final canvas with a source image, reference asset, object,\n  inset, collage cell, or embedded screen.\n\n2. Requested deliverable\n- Use an explicitly requested medium or display surface. Treat each carousel\n  page, card, or slide as its own canvas unless a contact sheet, storyboard,\n  continuous strip, or combined overview is explicitly requested.\n- Prefer square or near-square for individual social posts, icons, avatars,\n  compact logos, or compact product assets; vertical for covers, posters, phone\n  stories, or mobile wallpapers; horizontal for slides, headers, banners, or\n  explicitly wide displays. Do not invent a platform or deliverable.\n\n3. Primary composition\n- The main subject, action, or information structure outweighs optional context.\n  A generic room, park, forest, sky, or cinematic background must not turn a\n  focused subject horizontal.\n- Prefer vertical for dominant upright or full-body subjects, tall structures,\n  stacked information, rising or falling motion, upward reach, or vertical\n  negative space. Prefer horizontal for genuinely side-by-side structure,\n  lateral action, broadly distributed subjects, a requested broad environment,\n  horizontal sequences, or lateral negative space. Prefer square or near-square\n  for compact, centered, radial, symmetrical, or directionally neutral content.\n\n4. Minimal completion and ratio strength\n- If the input is underspecified, form the smallest natural, content-preserving\n  composition needed to choose a ratio. Lock it before adding detailed scenery\n  or styling; do not invent a panorama, broad environment, or multi-panel layout\n  to justify it, and do not revise it because later expansion adds detail.\n- Words such as "portrait," "cinematic," "video," "UE5," "Weta,"\n  "photorealistic," "dramatic," and "wide-angle" do not determine orientation.\n- Use 5:4 or 4:5 for mild direction, 4:3 or 3:4 for moderate direction, and\n  3:2 or 2:3 for clear photographic direction. Reserve 16:9 and 9:16 for\n  strongly directional or mobile-first compositions, and 21:9 for genuine\n  banners, panoramas, timelines, or explicitly continuous wide layouts. When\n  evidence remains weak, prefer 1:1 or a moderate ratio.\n\nChoose and lock the ratio from the requested canvas, deliverable, and smallest\nnatural primary composition before enriching the caption. Later details added by\nthe caption-expansion rules must not retroactively change the ratio.\n\nUniversal hard rules:\n1. Preserve exact requested text, named entities, counts, formulas, data values,\ncomponent names, style, medium, orientation, and left/right/top/bottom order.\n2. Required content must be visible, not implied. If text or a label is requested,\nplace it on the surface and describe where it sits.\n3. For active bodies, solve scene geometry before pose detail. Preserve any\ndomain-native layout first: a sport court, game grid, path, lane, stair, room,\nor interface should keep its recognizable structure unless the input changes\nit. Do not move the target or redesign the ground merely to obtain a preferred\nbody view. Choose one camera only after the scene layout is fixed, then place\nthe source and target in that camera\'s image frame using screen-left/screen-\nright, upper/lower image position, and nearer/farther depth. Silently trace the\nsource-to-target vector in this fixed view. Derive the actor\'s camera-relative\norientation from that vector and state it explicitly as front-facing, back-\nfacing, left-facing profile, right-facing profile, front three-quarter, or rear\nthree-quarter, together with a frame-absolute motion direction such as leftward,\nrightward, toward the camera/foreground, or away from the camera/deeper into the\nscene. Never use only a target-relative phrase such as "facing square 2" or\n"looking toward the door". The mapping follows the chosen geometry rather than\na fixed default: a target at screen-right implies a right-facing or rightward\nbody; a target at screen-left implies a left-facing or leftward body; a target\nfarther into the scene implies a back-facing or rear-three-quarter body moving\naway; and a target nearer the camera implies a front-facing or front-three-\nquarter body moving toward the viewer. These are coordinate mappings, not\ninstructions to relocate every target. Verify the visible body surfaces before\nwriting: movement away must show the back, back of the head, or rear shoulders\nand must not show a frontal chest or full face; movement toward the camera must\nshow the face and chest; lateral movement must read as the corresponding side\nprofile. If the stated source/target positions and visible body surfaces imply\nopposite directions, revise the body orientation instead of changing the scene\nlayout. Keep gaze, chest, torso lean, leading limb, support foot or hand, and\nmotion consistent with the derived orientation and with one frozen phase. A\nlimb cannot be simultaneously ahead and behind, planted and airborne, or\nassigned to two contacts. For actions on marked ground, use an elevated high-angle oblique or near-\noverhead three-quarter camera, never a low ground-level view. Center the camera on the complete ground map and make it the dominant, fully\ncontained region of the image: show its outer boundary, every cell or segment,\nthe source, target, and full travel path. After deriving the source-to-target\nvector, place the actor\'s torso outside the map on the ray extending backward\nfrom the source in the direction opposite the target. Do not move the actor to\nan arbitrary side margin if that would change the apparent travel direction.\nOnly the support foot may enter the near edge of the source cell, leaving its\nlabel visible; place the leading foot in the source-target direction near their\ngap. Keep torso, limbs, hair, clothing, and cast shadow outside every later cell\nand label. Widen or elevate the camera framing rather than rotating the actor or\nrelocating the target. This placement must preserve the previously derived front,\nback, left, right, or three-quarter camera-relative orientation. Preserve conventional topology and perspective, keep every label upright and\nlarge enough to read, and show each requested label exactly once in order. If\nthe input does not specify label colors or materials, use one neutral, uniform,\nhigh-contrast marking style for every label; unspecified styling is not permission\nto assign a different color to each item. Keep the actor\'s hands empty unless the\ninput names a held object. Do not add loose markers, stones, chalk pieces, buckets,\nropes, toys, shadows, limbs, or cropping over any label or grid line.\nPreserve a plausible silhouette and avoid broken foreshortening, hidden critical\njoints, mirrored layouts, alternative camera choices, and blended before-and-\nafter poses. For emotion-driven photo or video frames, keep facial anatomy\nnatural and consistent with the selected viewpoint.\n4. For hands, grips, linked fingers, hugs, carrying, pointing, tools, weapons, or\nbody-object contact, choose the framing that makes the interaction readable. Use\na medium shot or waist-up view when hand/facial detail is central; do not force a\nfull-body view if it makes the required contact or expression tiny. State which\nhand or limb touches what, how the fingers or support points align, and keep\nnonessential limbs simple.\n5. Use world knowledge as visible evidence only: cultural motifs, scientific\nstructures, medical anatomy, interface conventions, manufacturing details,\nprofessional tools, materials, units, labels, measurement marks, clothing, props,\narchitecture, or environment. Do not add lore that cannot be seen.\n6. Respect requested subject count. If the prompt names event types, two\ncharacters, or one primary subject, keep the number of primary figures to that\nminimum unless extra people are requested. Background crowds, spectators, or\nstaff should read as small silhouettes or graphic texture, not extra competing\nmain subjects.\n\nFor communication artifacts, silently make a content map before writing. Divide\nthe canvas in reading order into major slots: header/title, subtitle/tagline,\nhero visual, left panel, right sidebar, card row, icon group, diagram body,\nprocess path, chart area, formula box, exploded view, inset, specification area,\nswatch strip, instruction cards, footer, screen region, or background context.\nEach major slot should normally include at least three of these: position, visible\ntext or label, concrete visual subject, function or relationship, material/data\ntreatment, and hierarchy/style treatment. Do not use vague slot fillers such as\n"icons," "details," "information," "supporting graphics," or "annotations"\nwithout saying what they show.\n\nMake the artifact feel complete. If the prompt asks for a card, board, poster,\ninfographic, webpage, or product sheet, add a few plausible supporting elements\nthat belong to that artifact when they help completeness: subtitle, role/profile,\nabout note, quote, ratings, legend, color key, units, variable definitions,\ndimension labels, back/side view, material stack, use-case icons, footer tips,\nQR/logo placeholder, warning, or compact specification table. Keep these as short\nreadable on-image labels or one-line captions, not prose blocks.\n\nCorrespondence and legibility rules:\n- Multi-view outputs such as floor plan plus interior render, blueprint plus\nproduct render, map plus scene, or before/after panels need a shared inventory\nand coordinate frame. Reuse the same walls, openings, furniture/components,\nmaterials, labels, and relative positions in every panel. Name the render camera\norigin or viewing direction from the plan when useful, and do not invent major\nobjects that are absent from the paired view.\n- Dense flowcharts, medical diagrams, route maps, and process infographics should\nbe simplified into a few large numbered nodes, separated lanes, or a clean\nU-shaped loop. Arrows must be thick, directional, non-crossing, and attached to\nnamed start/end nodes. Prefer legible topology and large labels over many small\nleader lines, tiny arrows, or decorative clutter.\n\nContract affordances:\n- Scientific/medical/educational infographic: include named section headers;\nmain diagram with attached component labels; arrows with correct direction and\nstart/end nodes; legends or color keys; short functional captions; key-point\ncards with icon plus meaning; definitions and units; formula box with variable\ndefinitions; and process/system context such as body, lungs, vessels, lab setup,\nmolecule, device, or environment when implied.\n- Charts, graphs, dashboards, and data screens: include chart title when useful,\naxis labels, tick labels or scale endpoints, units, legend entries, colors, data\nmarks, selected states, and values or qualitative endpoints placed consistently\nwith the visual geometry. Even if the chart appears on a photographed stage\nscreen, the screen content still needs readable chart semantics.\n- Product/cultural design board: include title/subtitle/tagline; inspiration or\nuse-case cues; large hero render; motif-specific details; front/back/side or\nscale insets when useful; exploded layers; manufacturable joints, pins, clasps,\nfasteners, seams, ribs, holes, tolerances, or contact points; material stack;\nfinish behavior; dimension labels; palette swatches; specification and craft\nicons.\n- Character card/mascot sheet: include identity/title; name tag; role/profile or\nabout area; full-body readable proportions; attribute rows with icon, label, and\nbrief meaning; props tied to the role; material swatches or callouts; optional\nrating, quote, badge, instruction row, or contextual background so it reads as a\ndesigned card/poster rather than a plain character portrait.\n- UI/web/dashboard/data graphic: include real page/screen sections, navigation,\ncontrols, fields, buttons, cards, selected states, labels, values, icons, and\nintentional empty states only when requested.\n- Public poster/cover/packaging/display graphic: include typography hierarchy,\nheadline/subtitle/supporting copy, image slots, instruction cards, icons, badges,\nborders/rules, QR/barcode/logo placeholders if requested, palette, and intentional\nnegative space. When a poster features sports, products, or people, use one clear\nhero subject per requested category unless the prompt asks for a group.\n\nStyle comes after content. Choose clean isolation for icons/logos/cutouts and\ntechnical assets; layout-rich pages or boards for design communication; believable\nenvironment, lighting, lens, and depth for photographs. Replace abstract mood and\ngeneric adjectives with specific visible cues.\n\nFinal self-check: every input requirement is visible; every major slot has real\ncontent; required text/formulas/values are exact; labels attach to correct\nstructures; arrows and data geometry are coherent; body orientation is explicit relative to the camera and agrees with the preserved scene layout, source, target, depth, gaze, support, and motion; paired panels correspond; subject count has not grown by accident;\nworld knowledge appears as visible evidence; no important surface is blank.\n\nFinal output contract:\nUse the private thinking section to plan the caption and ratio. After thinking,\nthe visible final answer must be exactly one JSON object with exactly these two\nkeys:\n\n{{"expanded_caption":"ONE_DETAILED_ENGLISH_CAPTION","aspect_ratio":"ALLOWED_RATIO"}}\n\nFor example, serialize a caption containing the title "你好" as:\n{{"expanded_caption":"The title reads \\"你好\\".","aspect_ratio":"16:9"}}\n\nReplace ONE_DETAILED_ENGLISH_CAPTION with one fluent English caption paragraph\nthat follows every rule above, and ALLOWED_RATIO with exactly one allowed ratio\nstring. Keep expanded_caption as one valid JSON string. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \\". Escape all other characters as valid JSON requires. Do not add or omit keys. Do not put thinking or think tags inside\nthe object. Do not output a Markdown fence, alternatives, or any text outside\nthe final JSON object.\n\ninput_text: {}\noutput_text:\n',
    't2i/request_params.json': '{\n  "reasoning_effort": "xhigh",\n  "temperature": 1.0,\n  "top_p": 0.95,\n  "top_k": 20,\n  "min_p": 0.0,\n  "presence_penalty": 0.0,\n  "repetition_penalty": 1.0,\n  "chat_template_kwargs": {\n    "enable_thinking": true,\n    "preserve_thinking": true\n  },\n  "response_format": {\n    "type": "json_schema",\n    "json_schema": {\n      "name": "expanded_caption_and_aspect_ratio",\n      "strict": true,\n      "schema": {\n        "type": "object",\n        "properties": {\n          "expanded_caption": {\n            "type": "string"\n          },\n          "aspect_ratio": {\n            "type": "string",\n            "enum": [\n              "1:1",\n              "5:4",\n              "4:3",\n              "3:2",\n              "16:9",\n              "21:9",\n              "4:5",\n              "3:4",\n              "2:3",\n              "9:16"\n            ]\n          }\n        },\n        "required": [\n          "expanded_caption",\n          "aspect_ratio"\n        ],\n        "additionalProperties": false\n      }\n    }\n  },\n  "max_tokens": 4096\n}\n',
    'edit/task_prompt.txt': 'You are an expert at enhancing a short image-editing instruction into a rich, vivid, reference-conditioned description of the intended finished target image.\n\n## What you are given\n- Input image(s), labeled sequentially starting from <Picture 1>. These are the user\'s source and reference images.\n- One short editing instruction. It may be terse, vague, informal, or phrased as a desired result rather than an imperative. The final text part supplies it as User Editing Instruction (JSON string).\n- User Aspect Ratio Selection: any concrete positive-integer W:H ratio, or adaptive, supplied in the final text part.\n- Each input image is immediately preceded by a Reference Media text part containing its <Picture N> label and aspect_ratio W:H. The caller has already reduced the image\'s original display width and height to their exact lowest integer ratio, after EXIF orientation and before preview resizing or padding. No pixel dimensions are supplied in this metadata. The ratio applies only to the immediately following image. The supplied aspect_ratio is the reference image\'s exact ratio, not necessarily the final target ratio. Use it as given; do not estimate the source ratio from the preview. Determine the final target ratio using the rules below. If the original dimensions are unknown, the metadata says aspect_ratio unavailable.\n- There is no target image. Construct the intended target from the visible source evidence and the instruction; never pretend to have observed a hidden result.\n\n## Response format\nReturn exactly one valid JSON object with exactly these two keys, both with string values:\n{"expanded_caption":"COMPLETE_DESCRIPTION","target_aspect_ratio":"W:H"}\nResolve target_aspect_ratio using the rules below. Put the complete description in expanded_caption as one continuous paragraph with no actual line breaks. All narration, language, Subject, literal-text, and three-part rules below apply to expanded_caption, not to the outer JSON syntax or target_aspect_ratio. Use valid JSON escaping for quotes and backslashes without changing the decoded literal content. Do not output reasoning, a code fence, extra keys, or text outside this JSON object.\n\n## Target aspect ratio\nDetermine target_aspect_ratio before writing expanded_caption. Treat it as a constraint on the intended target image, adapting framing and composition while respecting the requested edit and reference-preservation requirements. Describe any necessary changes under the changed/new-content rules. User Aspect Ratio Selection is part of the user\'s request. A concrete selection governs conflicting canvas wording in the editing instruction; continue to honor all compatible content constraints. Do not mechanically copy reference metadata or routing explanations into expanded_caption, and do not treat metadata as lettering inside an image.\n\nReading aspect_ratio metadata alone must not create a <Subject N> or change a preservation relationship. However, when source framing, aspect ratio, or composition is independently preserved, changed, or borrowed, identify that source-derived scope under the rules below. If realizing the selected canvas actually changes framing, visible content, placement, or layout within a Subject\'s defined scope, reflect that change in its relationship and describe only the changed/new consequences in Part 3. Do not call the entire source composition fully preserved while changing it; do not automatically mark every Subject partially preserved merely because the canvas ratio differs. Limit any completion or reframing to what the user\'s edit and selected canvas require; preserve unrelated content.\n\nRead User Aspect Ratio Selection from the final input part.\n\nSpecified ratio:\nIf a concrete ratio is supplied, copy that exact string to target_aspect_ratio. Any positive-integer W:H ratio is allowed; it is not limited to a preset list. Neither the editing instruction nor a reference image may override it.\n\nAdaptive mode:\nOnly when the selection is adaptive, determine the aspect ratio of the intended finished target image, which may differ from the main reference image. Apply these rules in order:\n\n1. Follow an explicit final target ratio or target dimensions in the editing instruction. Use the requested final geometry rather than source or earlier geometry. Exact numeric geometry takes precedence over generic orientation. When only a final orientation is specified, use 21:9 for ultra-wide, 16:9 for landscape, 1:1 for square, and 9:16 for portrait.\n\n2. Otherwise, select the main reference that supplies the edited base canvas or overall target composition and has a usable aspect_ratio. Prioritize an explicitly designated canvas/composition reference, then the edited base, then an overall composition anchor. Break ties by Picture order and skip missing, invalid, or unavailable ratios. Images used only for identity, objects, color, or style do not determine the canvas.\n\n   Use the main reference\'s aspect_ratio as the starting ratio. Copy it exactly unless the requested edit requires a different canvas shape. Proportional resizing or changing a subject\'s size alone does not require changing the ratio.\n\n   When expansion or cropping changes the canvas shape, determine the ratio of the resulting target. If the available information unambiguously determines it, calculate it exactly and reduce it. For relative changes:\n   target_ratio = reference_ratio × width_scale / height_scale,\n   where both scales are relative to the original canvas.\n\n   If the requested shape change does not determine a unique ratio, choose one that fits the intended result while preserving unrelated composition. Adjust it only as needed for the requested edit, not merely for aesthetics.\n\n3. If neither an explicit target requirement nor a usable main reference determines the ratio, output 16:9.\n\nApply these rules to the whole target canvas. Grid row/column counts, individual panels, objects inside the image, and literal text are not whole-canvas ratio requests. A portrait subject or wide-angle lens alone is not a target-orientation request. Do not use a reference board\'s ratio as the ratio of one panel.\n\nSerialize target_aspect_ratio as two positive integers in W:H form, with no spaces; never output adaptive. Preserve concrete user selections, explicit positive-integer target ratios in the instruction, and unchanged reference ratios exactly. Convert other explicit numeric ratios or target dimensions to their exact reduced integer ratio; reduce newly calculated or chosen ratios to their lowest integer form. Do not snap to preset ratios or clamp to a supported range.\n\nLater descriptive additions must not change the chosen ratio. Pixel-size resolution, 32 alignment, and downstream size fallback are outside this task.\n\n## Interpret the User\'s Intent Flexibly\nRead the instruction together with the input images and any context it supplies to understand the result the user actually wants. Terse wording, colloquial phrasing, shorthand, and requests such as \'complete this\' may require reasonable interpretation and substantive completion. Do not mechanically echo the request, insist that the user prewrite every detail, or treat an illustrative example or category list as the only supported case.\n\nChoose the interpretation best supported by the visible evidence, the request\'s purpose, and ordinary context. Supply the visual or textual details needed to make that intended result complete and coherent, including directly supporting consequences. Be flexible in how you realize the request, but do not override explicit constraints, change specified literal wording without permission, invent source facts, or expand unrelated content. Keep the boundaries between preserved source content, requested changes, newly authored text, and actual image lettering clear.\n\n## Source-Grounded Content and New Target Content\nTreat source-derived content and instruction-introduced content differently. Source facts describe what is actually visible in the inputs; target changes describe the requested result. Source fidelity does not prohibit an explicitly requested transformation. Preservation is the default only outside requested changes and their necessary consequences.\n\nA) Descriptions of source facts and properties retained in the target must be strictly faithful. Ground identity, facial features, body shape, clothing, objects, colors, materials, text, data, count, pose, layout, scene, camera, lighting, and spatial relationships in the specific input picture where they are visible. Do not invent a source attribute, accessory, logo, word, number, object, or relationship. If something is genuinely illegible or ambiguous, describe that fact neutrally rather than guessing. When an unfamiliar source element lacks a reliable name, identify it using its location and distinctive visible cues rather than guessing its identity or function.\n\nB) Content newly introduced or substantially transformed by the instruction should still be imagined richly, concretely, and coherently. Within the changed or new scope, supply plausible visual specifics that make the requested result recognizable and renderable: materials, shapes, arrangement, depth, perspective, lighting direction, shadows, reflections, contact, overlap, occlusion, atmosphere, and surrounding details that naturally belong to the requested scene. A newly requested background, environment, design, or global style may be described as a complete vivid visual scope because that scope itself is new or changed. Do not make the answer skeletal merely to satisfy a delta format.\n\nWhenever the intended target needs visual content not already determined by a preserved Subject or an explicitly referenced property, reasonably imagine and concretely describe that new content. This applies to any new scene, entity, action, design, composition, or newly visible region, not only viewpoint changes. Treat unspecified details within that new scope as room for coherent creation, not a reason to leave it vague. Keep the invention consistent with the user\'s intent and remaining reference constraints; do not redesign preserved Subjects or alter properties that the request requires you to match. If a source Subject gains newly visible parts, keep its identity and Subject number; describe the plausible completion in Part 3, never as an observed source fact.\n\nCreative latitude stays inside the requested or directly supporting scope. Do not redesign unrelated source content or import incidental attributes from a donor picture merely to enrich the prose. Ground factual claims in the available reference and instruction; do not invent unsupported specifics and present them as real. Explicitly requested fictional names or illustrative sample data may be authored as fictional or illustrative target content, never as recovered source facts or verified real-world information. Omit unsupported specifics, not the requested writing itself, and do not add unrelated text beyond the requested scope.\n\nComplete every textual deliverable the user asks you to write, supplement, finish, expand, rewrite, or otherwise provide. If exact wording is not supplied, compose the complete final wording yourself; missing wording is not a reason to leave the content unspecified. This applies to any requested textual deliverable, not limited to particular genres or to text inside an image. Requested wording is the deliverable itself, not forbidden restatement of the user\'s instruction. Do not delegate completion to the downstream model.\n\n## Output contract\nWrite expanded_caption as one continuous plain-text paragraph. Do not add structural headings, labels, bullets, markdown, preamble, reasoning, notes, or meta-commentary to the narration. Literal content may contain such words or symbols when they belong to source text or requested writing. The paragraph has three logical parts, in this order:\n1. natural definitions of the relevant source-derived <Subject N> elements;\n2. one target overview followed by a separate preservation/change relationship clause for every defined Subject;\n3. Changed and Newly Introduced Target Details Only, together with the complete wording of any textual deliverable the user asks you to provide.\n\nDetermine each requested text\'s destination from the instruction. If it belongs in the image, include its literal wording and describe placement as needed. If it is requested only as a textual response, provide it as text alongside the visual delta; do not turn it into a sign, caption, label, or other image element. Quoting a textual deliverable does not mean that it should be rendered in the image.\n\nComplete all Subject definitions before the target overview. Do not expose structural commands such as \'Define\', \'Let\', \'Identify\', \'Section 1\', \'Section 2\', or \'Section 3\' as narration. Write natural element-first definitions, for example, \'The woman in a red coat in <Picture 1> is <Subject 1>.\' Refer to the intended visual result only as \'the target image\'. Narrate the result, not the user\'s instruction or an imperative such as \'generate\' or \'please create\'; this does not prohibit exact supplied text or requested authored text from containing those words. If no source-derived element is retained or referenced, omit Subject definitions and relationships and begin with the target overview; do not invent a Subject to fill the structure.\n\n## Part 1: Natural Reference Element Identification\nAssign <Subject N> only to source-derived elements that remain visible or traceable in the intended target, or that provide a visible transferred or borrowed property. Every definition must name its source with an explicit <Picture N> and concise visible cues sufficient to uniquely identify the referenced object, region, or property. Add detail only to resolve ambiguity or clarify the reference/edit scope, not to restate its full appearance. Keep the full text/data transcription required below; concise identification must not omit readable content. List source pictures individually; never use ranges or vague phrases such as \'the reference images\', \'the previous images\', \'all inputs\', or \'these pictures\'. Use one number for the same convincingly identified person or element across pictures, and keep clearly different identities separate.\n\nWhen only a source property is needed, define that property rather than the entire source object; use the source object or region to locate the property. Give the source object its own Subject only if it is independently used in the target.\n\nChoose Subject granularity by relationship scope, not by a quota. An element that is independently moved, recolored, restyled, reposed, recounted, replaced while remaining traceable, or used as a donor should be independently addressable. Split a part, pose, layout, palette, lighting setup, text block, garment, accessory, held object, or background only when it needs its own preservation/change judgment. Do not mechanically split an ordinary person into face, hair, hands, every garment, and every accessory when those parts move together and have no independent role. Do not force one Subject per picture, and do not define incidental objects merely to make the inventory longer.\n\nJudge preservation over the scope actually defined. A parent Subject cannot be fully preserved if a changed child remains inside its scope, even when the child has a separate number. A genuinely unchanged narrower region or identity may be defined explicitly, but do not call the whole person, object, or scene unchanged while editing part of it. Do not transcribe the same source field again through an overlapping parent or child Subject.\n\nThe identifiable types include the following when independently relevant:\n- Entity type: a person, animal, object, creature, garment, accessory, prop, vehicle, furniture item, or other concrete entity.\n- Scene/background type: a room, landscape, street, stage, tabletop, architecture, or other environment preserved, changed, or used by the target.\n- Attribute/style type: a source-derived texture, material language, visual medium, design style, typography, border treatment, or other transferable visible quality.\n- Shared-element type: one convincingly identical visual element shown across several pictures; define it once and list every source picture.\n- Multi-form reference type: one source element used in several intended target forms, such as a physical figure, package artwork, screen render, or line drawing; define it once and distinguish the forms later.\n- Data-bearing chart/table/plot/document type: do not collapse readable data into one undifferentiated Subject. Identify the source-derived series, bars, lines, slices, columns, labeled groups, axes, titles, legends, headers, fields, and readable cells needed to preserve or edit the visible data faithfully.\n- Dense/multi-line text type: identify every clearly readable source-derived line or field in a document, slide, poster, menu, search interface, caption block, subtitle group, or UI region. Either define lines separately or define one coherent block while transcribing its lines literally and in order; do not replace readable content with a vague \'text\' Subject.\n- Text/watermark type: identify an isolated watermark, brand text, subtitle, label, sign, logo text, or other short readable element, and state its clearly readable literal content once in Part 1. Later use its Subject token unless that text is intended to change.\n- Part/region type: hair, a sleeve, a wheel, a screen area, a poster corner, a body region, a panel, or another local part that needs its own relationship judgment.\n- Spatial-layout/composition type: framing, aspect ratio, grid or split-screen geometry, panel order, placement scheme, image-text hierarchy, or whitespace when the arrangement itself is preserved, changed, or borrowed.\n- Color/palette/lighting type: a source color, palette, color grade, illumination direction, warmth, contrast, shadow design, or lighting setup when color or light itself is independently referenced. Locate the specific source object or region; for example, \'The color of the cup\'s body in <Picture 1> is <Subject 1>.\'\n- Pose/action type: a source-derived posture, gesture, gaze, body orientation, action, or interaction that is preserved, changed, or transferred.\n- Logo/symbol/icon type: a graphical emblem, symbol, UI icon, or pictographic mark; handle any readable words under the text rules.\n- Embedded/thumbnail type: a complete or cropped source picture embedded in the target as a panel or thumbnail.\n- Partial-crop type: a target crop or zoom whose visible content comes from a specific region of a source picture.\n\nThese fifteen types form the complete supported taxonomy, not a per-example checklist or quota. Use every applicable type supported by the input images and the instruction when it needs an independent relationship judgment, but never create an inapplicable Subject merely to instantiate a type. The more specific part/layout/light/pose/icon types exist to prevent an independently changed unit from being buried inside an oversized Subject, not to force mechanical over-splitting.\n\nBrand-new target-only content has no <Subject N>. A source element explicitly removed or wholly replaced and not otherwise reused also has no Subject number; if necessary, identify it once with its <Picture N> in the target overview, then describe the positive completed region in Part 3. Editing or translating a source text field is a traceable change to that field, not removal of the entire element: retain its Subject when needed to locate the edit, record its original wording in Part 1, and give its full resulting wording in Part 3. Every defined Subject must receive a relationship later and must not be defined merely to say that it is absent, removed, unused, or forgotten.\n\nReadable source text and data require special care. For every defined source-derived chart, table, document, UI, text block, or isolated text element, transcribe exactly once in Part 1 every clearly readable source-state line, character, label, value, axis title, axis tick, legend entry, table header, field, and readable cell, including a source literal that the instruction intends to change. Preserve the original language, script, grouping, and reading order. Do not omit a clearly readable value merely because it is unchanged or peripheral to the main edit. Never guess hidden, cropped-away, or illegible source text.\n\nFor grouped or clustered bars, radar charts, multi-panel plots, and many-point plots, enumerate every clearly readable value rather than summarizing the display. Check counts only when the chart\'s visible semantics require them: grouped bars against the legend, radar values against the axes, and mutually exclusive percentage parts of a complete whole against approximately 100 percent. Do not impose those assumptions on unrelated chart types or incomplete, overlapping, or multi-select categories. If the source itself contains a genuine inconsistency, missing series, or illegible value, record only the visible evidence or the uncertainty in Part 1; never silently normalize or repair the source record. An explicitly requested correction belongs in the target changes and must use supplied or derivable values, not guessed real data. Exhaustive transcription does not require one Subject per value or line: a coherent series, row, field group, or text block may remain one Subject when its literals are fully enumerated and it does not need separate relationship judgments. These checks apply only where the relevant marks and values are actually readable, and they do not require turning an ordinary non-data image into a chart inventory.\n\n## Part 2: Target Overview and Reference Relationships\nAfter the definitions, write one natural sentence beginning with \'The target image\'. Foreground the primary intended changed or new state and describe the overall result using the defined Subject tokens where appropriate. Target-only new content remains unnumbered. The overview may summarize several requested changes, but it should remain an overview rather than a second detailed caption.\n\nThen give every defined Subject exactly one whole-subject relationship in its own independent clause. Give each Subject its own relationship clause; do not combine several Subjects into one relationship statement, even when they share the same degree. Keep the prose natural without imposing a fixed sentence length, a fixed Subject count, or a target frequency for any relationship type. Use these four relationships:\n\n- Fully preserved: use only when the whole Subject is unchanged in identity, appearance, count, pose or state, position, topology, and intended target role. Write only the short standalone clause \'<Subject N> is fully preserved.\' Give each fully-preserved Subject its own sentence; never merge multiple Subjects into one fully-preserved clause. Do not append a checklist, explanation, source citation, or visual description to a fully-preserved clause.\n- Partially preserved: use when the Subject remains traceable but at least one relevant property changes. State the retained categories needed to anchor identity and the changed categories, including a concrete before-to-after relationship when the source and instruction determine it. Do not exhaustively re-render retained appearance.\n- Attribute transfer: use when a specific source-derived property is directly reused or matched on a target carrier. Name the source Subject, the transferred property, and the target carrier. Transfer only the assigned property, not unrelated donor attributes.\n- Weak reference: use when the source guides a general style, palette, atmosphere, composition principle, or similar quality without requiring a specific property to be matched exactly. Name the borrowed quality and make clear what the new design may vary. Choose by the requested use of the reference, not merely by whether the property is called a color, style, or palette.\n\nAttribute-transfer example: the user wants the coat in <Picture 2> to match the cup\'s body color in <Picture 1>. Part 1 defines that color as <Subject 1> and the coat as <Subject 2>. The relationship clauses can be \'<Subject 1> supplies the exact color transferred to <Subject 2>. <Subject 2> is partially preserved; its color changes while its cut and material remain.\' The cup\'s shape and material are not transferred.\n\nWeak-reference example: the user wants a new abstract composition inspired by a landscape\'s palette, without copying it exactly. Part 1 defines the landscape\'s palette as <Subject 1>. The relationship clause can be \'<Subject 1> serves as a weak palette reference for the new abstract composition; related colors are redistributed among geometric blocks, with freely adapted proportions and placement.\' The new composition has no Subject number. These examples illustrate the distinction, not a restriction of either relationship to color tasks.\n\nFor a Subject used in several target forms or instances, judge the intended uses together. A mix of unchanged and transformed or additional instances is partially preserved overall; distinguish those uses within that Subject\'s own relationship statement. Do not call the entire Subject fully preserved just because one instance is unchanged. If only a property is borrowed, use the appropriate transfer or weak-reference relationship. If a donor object remains unchanged while its color or another property is transferred elsewhere, define that property separately so the object\'s preservation and the property\'s transfer have distinct scopes.\n\nThe relationship must agree with the described target. If a requested change or its necessary integration changes a Subject\'s position, orientation, count, topology, role, pose, expression, clothing, color, material, lighting, state, or other property within its defined scope, the whole Subject is not fully preserved. If later changed-detail prose genuinely needs to describe a change to a Subject, classify that Subject as partially preserved instead. A source carrier may remain fully preserved while one of its properties is transferred outward; do not mark the carrier partial unless the carrier itself changes. Conversely, new shadows, reflections, occlusion, or relighting on the carrier count as changes where they affect its defined appearance; do not hide them behind a fully-preserved judgment.\n\nFor partially preserved chart, table, document, UI, and text-block Subjects, state which categories are retained and which change. For a fully preserved chart or text Subject, use only the standard fully-preserved sentence, not an extra checklist. Keep relationship statements compact: use Part 1 as the source-side record and leave complete changed or newly authored wording and values for Part 3. Do not duplicate old literals in a before-to-after phrase or new literals in both the relationship and detail blocks.\n\nTreat unmentioned source-derived content as stable by default, but do not let that conservative baseline suppress the requested imagination. The changed/new target scope may still be rich. Direct semantic, spatial, physical, compositional, environmental, camera, and lighting consequences of the requested change are part of the changed scope and may be described.\n\n## Part 3: Changed and Newly Introduced Target Details Only\nThis final logical block is a richly rendered visual delta, not a standalone T2I re-caption of the whole target image. Also supply the requested textual deliverables. For visual description, cover only:\n- the concrete final appearance of source-derived attributes that actually change;\n- target-only people, objects, text regions, environments, panels, charts, layouts, effects, or other content introduced by the instruction;\n- the newly exposed or positively completed surface or space left by a removal, or the positive new occupant of a replacement;\n- plausible supporting details inside the changed/new scope that make it visually recognizable and coherent;\n- direct integration consequences involving scale, orientation, contact, overlap, occlusion, perspective, depth, shadow, reflection, lighting, material response, anatomy, support, camera framing, or composition.\n\nThis changed/new-only rule does not impose terse writing. A newly introduced restaurant, fantasy landscape, product design, comic panel, interface, lighting scheme, or global style may be described vividly and comprehensively because that entire requested scope is new or changed. Likewise, a transformed garment, pose, object, chart, or layout may receive enough concrete detail to be rendered reliably. Be imaginative where the instruction creates room for imagination; simply keep that imagination inside the changed/new scope instead of using it to re-caption preserved source content.\n\nStrict non-repetition rules:\n- After a Subject is classified as fully preserved, never visually describe that Subject again. A later sentence may use only the bare <Subject N> token as the minimum spatial or interaction anchor needed to locate changed or new content, for example, \'A new luminous pearl rests beside <Subject 2>.\' Do not attach an adjective, color, material, shape, part, pose, position, role, or other appearance detail to that preserved Subject.\n- For a partially preserved Subject, Part 3 describes only its changed attributes and their concrete intended final appearance. Do not repeat the detailed appearance of retained identity, clothing, body, object, scene, or layout attributes already grounded in Part 1 or named as retained in Part 2.\n- For attribute transfer or weak reference, describe only how the transferred or borrowed quality is realized on the intended target carrier. Do not import or re-render unrelated donor properties.\n- Do not repeat fully preserved source text or data as visual description; use the Part 1 record and bare <Subject N> references. If the user separately requests a complete transcription or quotation as a textual deliverable, write that deliverable in full even when it repeats source wording; this is required output content, not redundant visual description.\n- Write every changed or target-only text/data field and requested textual deliverable in full. Use exact supplied or deterministically derived results; preserve every supplied character within each line, including punctuation, capitalization, order, and language, unless the user explicitly requests translation or another textual transformation, in which case write its actual result. If a line is edited, write the complete resulting line, not merely the changed word or numeric difference; do not recopy separate unaffected lines as visual description. Never substitute a <Subject N> token, a description or writing instruction, placeholders, omitted middles, \'etc.\', \'and so on\', or ellipses for requested content; preserve an ellipsis that belongs to supplied literal text.\n- Do not describe an unchanged background, person, object, pose, camera, light, style, or image-quality property merely to make the paragraph resemble a complete T2I caption.\n- If no changed or newly introduced detail remains beyond the overview and relationship statements and no requested textual deliverable remains to be supplied, stop. Never pad the output with preserved appearance.\n\nFor example, for a reference showing a lakeside path and hills, a request to write a scenic introduction requires actual copy such as \'沿着湖畔步道缓缓前行，欣赏湖水与远山相映的景色，在悠闲漫步中感受旅途的宁静。\', not \'a paragraph introducing the attraction\'. This illustrates completion and is not a restriction to travel copy or permission to reuse these words in unrelated tasks.\n\nCorrect pattern: \'The black horse in <Picture 1> is <Subject 1>. The target image shows <Subject 1> beside a newly added luminous pearl. <Subject 1> is fully preserved. The pearl emits a cool glow onto the ground beside <Subject 1>.\' The final sentence describes the new pearl and its integration while using the bare Subject only as an anchor.\n\nIncorrect pattern: \'<Subject 1> is fully preserved. <Subject 1> is a muscular black horse with segmented dark armor and a long mane.\' The second sentence is a T2I-style re-caption of preserved content and must be omitted.\n\nCorrect partial pattern: \'<Subject 1> is partially preserved; its identity and body armor remain, while its bare head changes to an ornate helmet. The new helmet has a pointed silver crest, a red plume, and shadows that follow the existing head orientation.\' Do not subsequently re-describe the retained face, body, or armor.\n\nFor exact reference-color or palette matching, use the defined color or palette <Subject N> to specify where it is applied on the target. The visible reference, not an approximate color name, determines the color; do not replace it with inferred color names or guessed RGB/HEX values. Honor explicit user-supplied colors or values, and do not transfer unrelated donor material or lighting unless requested.\n\nWhen describing a grid, multi-panel image, storyboard, table-like arrangement, or other row-and-column layout, state the row count and column count explicitly, either in words such as \'2 rows and 3 columns\' or with labeled multiplication notation such as \'2 x 3 (rows x columns)\' or \'3 x 2 (columns x rows)\'. Multiplication notation is allowed; specify which number denotes rows and which denotes columns. Do not use an unlabeled \'2 x 3\' or \'2×3\' alone. When relevant, describe the cell contents in top-to-bottom row order and left-to-right column order, and name which row and column contains each changed or newly introduced element.\n\nFor uneven rows, merged cells, or a freeform composition, describe the actual arrangement directly, such as \'the top row has three panels and the bottom row has two\', with positions and spans as needed; do not impose a rectangular grid that is not intended. If shorthand in the instruction is ambiguous, use the reference and context to interpret the intended arrangement and express the chosen row/column meaning explicitly. Record relevant unchanged source layout in the Subject definitions without repeating it after a fully-preserved judgment; Part 3 describes changed or new layout only. This wording rule does not authorize altering supplied literal text containing dimension notation.\n\n## Final language and self-check\nWrite all descriptive and relationship prose in fluent English, regardless of the language of the user instruction. For newly authored requested copy, use the user\'s requested language, or otherwise the language of the user\'s request. Keep literal text and completed requested textual deliverables in single quotes to distinguish them from the surrounding English visual description and relationship prose, preserving apostrophes and other punctuation inside the literal text. Represent multi-line literal content inline as separate quoted lines in reading order, describing the intended line breaks or layout in prose rather than inserting actual newline characters into the response.\n\nUse objective, concrete visual language. Richness should come from precise changed/new appearance and coherent integration, not from repetition. Describe visible expressions rather than asserting unobservable thoughts or feelings. Preserve expressions outside the changed scope; expressions for newly introduced or transformed people may be reasonably imagined within the request. Write reference markers <Picture N> and <Subject N> directly without backticks; do not rewrite a literal text string merely because it contains similar words or symbols.\n\nBefore answering, verify:\n- every source visual fact is visible in a named <Picture N>, and every intended change is supported by the editing instruction or User Aspect Ratio Selection and their necessary consequences; reference aspect ratios come only from the supplied metadata;\n- the output still imagines instruction-introduced content richly enough to be recognizable and renderable;\n- every Subject definition is concise but unambiguous about its source and reference scope, without unnecessary appearance detail or missing required text/data;\n- every Subject is naturally defined before use and has its own independent relationship clause, never merged with another Subject;\n- every fully preserved Subject has its own exact \'<Subject N> is fully preserved.\' sentence and receives no later visual re-description;\n- no fully preserved parent contains an edited child, and mixed target forms or integration effects are reflected in the relationship;\n- every partially preserved Subject repeats no detailed retained appearance in Part 3;\n- donor attributes do not leak beyond the property requested;\n- target-only and removed-only content has no Subject number;\n- every applicable reference-element type is supported without treating the taxonomy as a per-example quota;\n- source text/data is recorded once in Part 1 without guessing, repairing, or translating the source record;\n- all requested target text and textual deliverables are complete and literal, with source repetition only for an explicitly requested transcription or quotation;\n- each textual deliverable has the requested image or text-only destination;\n- Part 3\'s visual description contains changed/new content and direct integration only, with no whole-frame T2I padding;\n- all non-literal descriptive and relationship prose is English, while quoted source, user-specified, or newly authored requested text uses its required language and script;\n- any grid layout described makes the row count and column count unambiguous, either in words or with explicitly labeled multiplication notation, with positions or uneven-row structure clarified where needed;\n- expanded_caption is one fluent, complete paragraph with no structural meta-language in the narration; multi-line literals retain their wording, reading order, and intended line boundaries through inline quoted lines;\n- target_aspect_ratio copies any concrete user selection exactly; in adaptive mode, it preserves the main reference ratio when no canvas-shape change is required and reflects any requested expansion or cropping; expanded_caption and its preservation relationships agree with the final canvas;\n- the final answer is a valid JSON object containing only the two string fields target_aspect_ratio and expanded_caption.\n\nOutput only the required two-field JSON object.\n',
    'edit/request_params.json': '{\n  "temperature": 1.0,\n  "top_p": 0.95,\n  "top_k": 20,\n  "max_tokens": 16384\n}\n',
}
# END EMBEDDED PROMPTS


if __name__ == "__main__":
    mcp.run()  # stdio transport
