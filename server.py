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
  IMAGE_OUTPUT_DIR   where to save images, default: <script dir>/generated
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
OUTPUT_DIR = Path(
    os.environ.get("IMAGE_OUTPUT_DIR", str(SCRIPT_DIR / "generated"))
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
        "prompts. Use expand_prompt when the request is vague - a short idea, "
        "or an edit that doesn't say where/what exactly should change - "
        "especially for edits, where an unspecified change risks a literal "
        "misread. Skip it when the prompt is already fully specified; "
        "hand-written specifics work at least as well and keep control. Follow "
        "its recommended aspect ratio when the user didn't pick one; for edits "
        "you can pass its caption verbatim (<Picture 1>/<Subject 1> "
        "placeholders are the official format) or tighten it first - the "
        "expanded caption is an editable spec. Every result comes back as an "
        "inline image you can see: review "
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
    if not path.is_file():
        raise ToolError(f"Prompt-expansion file missing: {path}")
    return path.read_text(encoding="utf-8")


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
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


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
        prompt: What the image should show. Be specific and complete -
            subject, style, composition, lighting, colors, mood.
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

    Use this when the request is vague - a short idea, or an edit that doesn't
    say where/what exactly should change (especially edits: an unspecified
    change risks a literal misread). Skip it when the prompt is already fully
    specified - hand-written specifics work at least as well and keep control.
    Follow its recommended aspect ratio when the user didn't pick one.

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
        f"pe_prompts_dir: {PROMPTS_DIR}",
    ])


if __name__ == "__main__":
    mcp.run()  # stdio transport
