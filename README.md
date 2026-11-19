# image-mcp

Single-file MCP server for image **generation and editing** through a LiteLLM
proxy, built for `minimax-h3-image` (works with any OpenAI-images-compatible
backend: `/v1/images/generations` + `/v1/images/edits`).

Made for a vision-capable agent workflow: every tool returns the saved file
path **and the image itself inline** (a downscaled preview so provider image
limits aren't hit — the file on disk is full-res), so the agent can look at
the result right away, judge it, and iterate with `edit_image` until it
matches.

## Tools

| Tool | Does |
|---|---|
| `generate_image(prompt, size?, n?)` | Text → image(s) at `size` (e.g. `"16:9"`, `"2048x2048"`), saved under `generated/` |
| `edit_image(prompt, image_paths, size?, n?)` | 1–16 image files + instruction → edited image(s). Reference inputs in the prompt as "image 1", "image 2", … |
| `get_image(path)` | Re-serve a saved image (e.g. after context compaction) |
| `list_images(limit?)` | Recent images, newest first (path, size, timestamp) - the agent's memory across context compaction |
| `scale_image(path, width?, height?, factor?, fmt?, quality?, max_mb?)` | Resize any local image up/down with Lanczos and/or recompress - exact pixels, instant, no generation cost; optional png/jpg/webp output, quality knob or max-file-size target |
| `get_config()` | Report the active configuration (API key masked) - verify env setup without touching files |
| `expand_prompt(instruction, image_paths?, aspect_ratio?)` | Official MiniMax PE prompts: short idea → detailed caption + recommended ratio (T2I), or edit instruction + references → target description + target ratio (edit mode) |

Provider errors (HTTP status + body, e.g. content-safety `code 1027`) are
surfaced verbatim to the agent so it can rephrase and retry. No automatic
retries — H3 bills generation attempts even when the client times out.

## Sizes

`size` takes two forms:

- **Aspect ratio** `"W:H"` (recommended) — mapped to ~16 MP native resolution,
  snapped to multiples of 32: `1:1` → 4096x4096, `16:9` → 5472x3072,
  `21:9` → 6272x2688. Any ratio from 1:3 to 3:1 works.
- **Exact pixels** `"WxH"` — passed through as-is (multiples of 32 are safest).

Documented sweet spot: ~4–20 MP (native ≈ 16 MP). Out-of-envelope values are
still sent to the API, with a warning in the tool output — backends differ in
strictness (this stack is vLLM behind LiteLLM). Check what yours honors with:

```
uv run tests/live_size_check.py "16:9"
```

It prints requested vs. actual PNG dimensions. Full-res files can be large
(a 16 MP PNG is ~50 MB); the inline preview the agent sees stays small
regardless, and `edit_image` always uploads the full-res file.

## Prompt expansion (PE)

`expand_prompt` runs MiniMax's official prompt-expansion prompts, vendored
under `prompts/` and SHA-256-verified against the release manifest of
[MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE):

- **Text-to-image**: short idea → detailed caption + recommended aspect ratio
  (10-ratio enum → pass it as `size=`). Works on any chat model behind the
  proxy via `PROMPT_LLM_MODEL`.
- **Image editing** (when `image_paths` is given): edit instruction + reference
  images → complete target description + target ratio. Needs a vision-capable
  model (`PROMPT_VLM_MODEL`, defaults to `PROMPT_LLM_MODEL`). MiniMax marks
  edit-PE as under development (prompt only).

Model choice (measured on this stack, same instructions): `glm-5.3-flash`
expands ~2x faster (~10–13 s vs ~24 s) with comparable caption quality —
even slightly better caption→image traceability — and it accepts image input,
so one model covers both modes (hence the default). `qwen3.8-27b` is a
text-only fallback (slower). Check any model yourself:

```
uv run tests/compare_pe_models.py qwen3.8-27b glm-5.3-flash
uv run tests/check_vision.py glm-5.3-flash
```

Reference images are sent as downscaled JPEG previews with their *original*
aspect ratio in the metadata — exactly what the official pipeline prescribes.

Edit-mode expansion needs a large completion budget: reasoning models spend
most of their tokens on hidden thinking before the JSON appears (measured:
> 11K reasoning tokens for a single edit). `prompts/edit/request_params.json`
defaults to `max_tokens: 16384` — tune it there if an expansion is truncated.
`reasoning_effort` / `chat_template_kwargs` are forwarded to the proxy when
present in a mode's `request_params.json`.

For maximum fidelity, serve the actual fine-tuned expander on your own vLLM
(`vllm serve MiniMaxAI/MiniMax-H3-Image-PE`), put it behind LiteLLM, and set
`PROMPT_LLM_MODEL=MiniMax-H3-Image-PE` — no code changes needed. The official
sampling parameters (including reasoning settings for the PE model) live in
`prompts/t2i/request_params.json`.

### Reproducibility (seed)

`generate_image(..., seed=1234)` is **honored by this backend** (verified with
`uv run tests/probe_seed.py`: same prompt + size + seed → byte-identical
image). Use it to hold a composition steady while tweaking the prompt, or to
re-roll honestly with a new seed. If the backend changes, re-run the probe.

## Multi-reference editing patterns

`edit_image` accepts 1-16 reference images; the prompt refers to them by
position ("image 1", "image 2", ...) in the order given. The PE edit prompt
is built for reference-aware descriptions, so these recipes work well:

- **Character/object consistency**: generate a clean reference of the
  character once, then always pass it as image 1: "Put the character from
  image 1 into a rainy neon street; keep face, hair and outfit unchanged".
- **Attribute transfer**: "Color the car in image 1 exactly like the cup in
  image 2" - the expansion separates the donor property from the target.
- **Combining images**: "Replace the background of image 1 with the
  environment from image 2, keep the subject untouched".
- **Loose style transfer**: "Redraw the sketch in image 1 as an oil painting,
  using the palette and brushwork of image 2 as a loose reference only".
- **Turnarounds**: pass a front view and ask for the back or side view while
  keeping the reference for identity.

`expand_prompt(instruction, image_paths=[...])` expands these too - it takes
the same reference images and returns the complete target description.

## Resizing (no generation)

`scale_image(path, width?, height?, factor?, fmt?, quality?, max_mb?)` resizes
any local image with Lanczos and/or recompresses it - instant, free,
deterministic: downscale to chat-friendly copies or thumbnails, upscale onto
an exact canvas (e.g. wallpaper sizes), convert format (png/jpg/webp), or
shrink file size. Give width and/or height (the other side keeps the aspect
ratio), a factor like `0.5` / `2.0`, or compression alone (re-encode at the
current size): `quality=1-100` for jpg/webp (default 95), or `max_mb=2.0` to
auto-pick the highest quality under a size target (a PNG source is re-encoded
as jpg unless you set `fmt` - PNG cannot hit size targets predictably).

For *detail-true* enlargement of a generated image, prefer a re-render:
`edit_image(path, size="WxH")` re-synthesizes the scene at the target
resolution. The trade-off: Lanczos adds pixels, not detail; the re-render adds
detail, not fidelity (tiny text can drift). Pick per job.

## Run

With [uv](https://docs.astral.sh/uv/) installed, dependencies resolve
automatically on first start (pinned to Python 3.12 via `.python-version`):

```
uv run server.py
```

Without uv:

```
pip install -r requirements.txt
python server.py
```

## Configure

All settings are plain environment variables - nothing needs to be edited on
disk. Two ways, freely mixable (real environment variables always win):

**Option A (recommended): set them in the MCP client config.** No files to
edit - the env block travels with the config when you share it:

```json
{
  "mcpServers": {
    "minimax-image": {
      "command": "uv",
      "args": ["run", "C:\\path\\to\\image-mcp\\server.py"],
      "env": {
        "LITELLM_BASE_URL": "https://litellm.example.com",
        "LITELLM_API_KEY": "sk-...",
        "PROMPT_LLM_MODEL": "glm-5.3-flash"
      }
    }
  }
}
```

**Option B: a `.env` file** next to `server.py` (`KEY=VALUE` per line,
`#` comments). Handy when a client makes env editing awkward.

To verify what the server actually picked up, ask the agent to call
`get_config` - it reports every effective value and never includes the API key.

| Variable | Default | Meaning |
|---|---|---|
| `LITELLM_BASE_URL` | — (required) | e.g. `https://litellm.example.com` (add `/v1` yourself only if you want) |
| `LITELLM_API_KEY` | — | Bearer token, if the proxy needs one |
| `IMAGE_MODEL` | `minimax-h3-image` | Model name as configured in LiteLLM |
| `IMAGE_OUTPUT_DIR` | `<script dir>/generated` | Where PNGs are saved |
| `IMAGE_TIMEOUT` | `600` | HTTP timeout in seconds (H3 can be slow) |
| `IMAGE_PREVIEW_MAX` | `1024` | Max side (px) of the inline preview the agent sees; `0` = inline full-res. Full-res files are always saved to disk |
| `PROMPT_LLM_MODEL` | — (required for `expand_prompt`) | Chat model on the same proxy used for prompt expansion; must be vision-capable if you pass `image_paths` |
| `PROMPT_VLM_MODEL` | = `PROMPT_LLM_MODEL` | Optional separate vision model for edit-mode expansion |
| `PROMPT_TIMEOUT` | `1200` | Chat timeout in seconds (expansion models think for a while) |
| `IMAGE_HTTP_PORT` | `18090` | Localhost port of the built-in image host used for chat-renderable markdown links; `0` = random port, `-1` = disable |

## Plug into a client

CherryStudio (Settings → MCP → add JSON), same shape works for Claude Code & co:

```json
{
  "mcpServers": {
    "minimax-image": {
      "command": "uv",
      "args": ["run", "C:\\path\\to\\image-mcp\\server.py"]
    }
  }
}
```

Config (`.env`) and output dir are resolved relative to `server.py`, so the
client's working directory doesn't matter. If the client can't find `uv` on
its PATH, use the pip route and `command: "python"`.

Set the client's MCP tool timeout generously (≥ 5 min) — image generation can
take minutes.

## Use directly from GitHub (no clone)

`server.py` carries its dependencies as inline script metadata, so `uv` can
fetch and run it straight from the repo - no clone, no local copy (verified:
uv resolves the inline deps and runs the remote file):

```json
{
  "mcpServers": {
    "minimax-image": {
      "command": "uv",
      "args": ["run", "https://raw.githubusercontent.com/StackOverBuffer/image-mcp/main/server.py"],
      "env": {
        "LITELLM_BASE_URL": "https://litellm.example.com",
        "LITELLM_API_KEY": "sk-...",
        "IMAGE_OUTPUT_DIR": "C:\\path\\to\\generated",
        "PROMPT_LLM_MODEL": "glm-5.3-flash"
      }
    }
  }
}
```

Three things differ from a local install:

- **`IMAGE_OUTPUT_DIR` is required.** The default output dir is
  `<script dir>/generated`, and the script dir here is uv's temp directory -
  without this env var, images land somewhere ephemeral.
- **No `.env` discovery.** The fetched script runs from a temp dir, so all
  configuration must come through the env block (Option A above).
- **`main` is a moving target.** Every start runs the current branch tip.
  Pin a tag or commit for stability:
  `.../StackOverBuffer/image-mcp/v1.0.0/server.py`.

The dependency environment is built once and cached by uv afterwards; each
start only re-fetches the single script file.

## Agent usage pattern (optional)

The MCP already carries this policy in its tool descriptions, so a plain agent
follows it without any system prompt. Paste this into the agent's system prompt
only if you want the behavior spelled out explicitly (e.g. for weaker models):

```
For image requests: if the idea is vague - a short idea, or an edit that
doesn't say where/what exactly should change (especially edits: an
unspecified change risks a literal misread) - call expand_prompt first and
use its expanded caption and recommended size. Skip it when the prompt is
already fully specified: hand-written specifics work at least as well and
keep control. Then call generate_image; the tool returns the image inline -
review it against the user's request.

Judge HOW the result is off, then pick the cheapest fix that can actually work:
- Wrong composition, wrong subject, wrong overall concept, or the request was
  misunderstood: do not edit a bad base. Regenerate with a sharpened prompt
  (rewrite the caption yourself, or expand_prompt again with the failing part
  made explicit). For genuinely uncertain requests, generate two candidates
  (n=2) and keep the better one as the base.
- Right composition and subject, but wrong details, colors, style, or small
  elements: edit_image with a concrete, targeted instruction (reference
  inputs as "image 1"). Editing preserves what is already right.
- Two attempts share the same flaw: the prompt is the problem, not the model.
  Rewrite it (be more explicit about exactly that part) instead of rolling
  again.

Iterate until the image matches, then give the user the final path. When a
tool result contains markdown image links, include them in your reply so the
user sees the image directly in the chat.
```

## Showing images in the chat window

Images reach the chat through three channels:

1. **Inline image blocks** - every tool returns the image itself as an MCP
   image block. Recent Cherry Studio versions render these (and they are what
   vision models see); some builds show them only as a file card instead.
2. **Markdown links in the agent's reply** - the most reliable display path:
   the server runs a tiny read-only image host on `127.0.0.1`
   (`IMAGE_HTTP_PORT`, default 18090; `0` = random port, `-1` = disable) and
   every generation/edit result ends with ready-to-paste markdown plus the
   instruction to include it in the reply. The chat renders the image inline.
   The host serves only files under the output dir, only on localhost, behind
   a per-session random token, with `Cache-Control: no-store` so updated
   images always show fresh.
3. **File paths** - the full-res files stay on disk under the output dir;
   open them directly anytime (the file card's "Open with" works too).

If several server instances run at once (e.g. another agent or a test), the
first takes the configured port and the rest fall back to random ports -
each instance's links always carry its own port, so they never cross wires.

## Copying the server elsewhere

The minimal working bundle is:

```
server.py
prompts/            # PE prompt files (needed only for expand_prompt)
.python-version     # pins Python 3.12 for uv (recommended; avoids wheel gaps on very new Pythons)
```

Copy those, add the configuration as environment variables in the new
client's MCP config (`LITELLM_BASE_URL`, `LITELLM_API_KEY`,
`PROMPT_LLM_MODEL`, optional overrides) and start with `uv run server.py` -
`uv` resolves all dependencies automatically on first start. A `.env` file
works instead of env vars if preferred.

Without `prompts/`, everything except `expand_prompt` still works (its tools
fail with a clear error). Nothing else is needed: `tests/`, `generated/`,
`README.md` are development-time only.

## Verify

```
uv run tests/test_smoke.py
```

Boots a fake LiteLLM image API locally and checks: tool listing, generation
(n=2), size handling (ratio anchors, WxH passthrough, computed ratios, size on
edits, invalid input guidance), multi-image edit multipart contract, verbatim
provider error surfacing, get_image, preview downscaling, and a full stdio
round-trip. Test images land in `generated-test/` (safe to delete).

Two more scripts run against the real server (they cost API credits):

```
uv run tests/live_test.py               # generate -> review -> edit loop
uv run tests/live_size_check.py "16:9"  # verify the backend honors size
uv run tests/live_expand_check.py       # expand a short idea, generate from it
uv run tests/probe_seed.py              # verify the backend honors seed
uv run tests/compare_pe_models.py       # A/B expansion models (speed + captions)
uv run tests/check_vision.py            # which chat models accept images
uv run tests/list_models.py             # list models on the proxy
```

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE). Note AGPL's
network-use clause: if you serve a modified version of this server, the
modified source must be made available to its users.
