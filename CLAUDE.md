# CLAUDE.md

Guidance for coding agents working in this repository. `README.md` is the
user-facing documentation; this file is for agents changing the code.

## Working agreement (read first)

- When the user asks a question, answer it. Do not start editing code, adding
  features, or fixing anything unless the user explicitly asks for that
  change. "Would X be useful?" is a request for analysis and a proposal - not
  a build order. Propose, wait for the explicit go-ahead, then build.

## What this is

A single-file MCP server (`server.py`, stdio, FastMCP 4.x) exposing image
generation/editing through a LiteLLM proxy to an OpenAI-images-compatible
backend (MiniMax-H3-Image on vLLM). Built for a vision-capable agent loop:
every image tool returns saved file paths AND the image itself inline, so the
agent reviews each result and iterates (regenerate vs. edit decision policy
lives in the tool docstrings).

## Layout

- `server.py` - the entire server. No other runtime code exists; keep it that
  way ("as simple as possible" is an explicit user requirement).
- `prompts/` - vendored official MiniMax-H3-Image-PE prompts (t2i + edit),
  SHA-256-tracked in `prompts/manifest.json`. Note:
  `prompts/edit/request_params.json` is a LOCAL addition (flagged in the
  manifest), not part of the upstream release. Prompt/params files are read
  per call; code changes require an MCP client restart to take effect.
- `tests/test_smoke.py` - mock-API suite; the gate for every change.
- `tests/live_*.py`, `probe_*.py`, `compare_pe_models.py`, `check_vision.py`,
  `list_models.py` - live probes against the real server. They cost API
  credits; never run them as part of routine verification unless asked.
- `generated/` - user's image output. Never delete or reorganize it.
- `.env` - config fallback (gitignored, contains the API key; never print it).

## Commands (PowerShell - Windows box, there is no bash)

    uv run tests/test_smoke.py              # the gate; run after every server change
    uv run server.py                        # start the MCP server (stdio)
    uv run tests/live_size_check.py "16:9"  # example live probe (costs credits)

PowerShell gotchas:
- Don't append `2>&1` to native commands when you need a truthful exit code -
  it fabricates exit code 1 plus NativeCommandError noise. Check `$LASTEXITCODE`.
- PS 5.1: iterate JSON object properties via `.PSObject.Properties`, not
  `GetEnumerator()`.

## Hard constraints

- stdout belongs to the MCP stdio transport. `server.py` must never print to
  stdout; diagnostics go to stderr only.
- `.python-version` pins 3.12 on purpose: Python 3.15 has no pyyaml wheel and
  needs an MSVC toolchain to build. Do not bump it.
- Never log or echo the API key. `get_config` masks it (length only) - keep
  that guarantee when touching that tool.
- No automatic retries of image API calls: H3 bills attempts even when the
  client times out or disconnects. Surface provider errors verbatim
  (ToolError with status + body) instead.
- The localhost image host binds `127.0.0.1` only, serves only files under the
  output dir behind a per-session random token, and sets
  `allow_reuse_address = False`. That last one matters: on Windows,
  SO_REUSEADDR lets a second process silently co-bind a listening port, so all
  connections would keep going to the first instance. A busy port must fail
  the bind and fall back to a random one.
- Inline previews exist because full-res images are 12-50 MB PNGs. Never put
  raw full-res bytes into tool results (`IMAGE_PREVIEW_MAX=0` is opt-in only).

## Backend facts (learned empirically; re-verify with the matching probe if the backend changes)

- `size` accepts `"W:H"` (mapped to ~16 MP native, snapped to /32 - reproduces
  the 4096x4096, 5472x3072, 6272x2688 anchors exactly) and `"WxH"`
  (passthrough). The backend honors `size` (live_size_check).
- `seed` is honored: same prompt + size + seed gives byte-identical output
  (probe_seed).
- Edits are multipart: single file -> field `image`, multiple -> `image[]`.
- Edit-mode expansion needs a large completion budget: the configured expander
  (glm-5.3-flash) is a reasoning model and spent >11K tokens on hidden
  thinking for one edit. `prompts/edit/request_params.json` defaults to
  `max_tokens: 16384`; T2I uses the official 4096 from its params file.
- The proxy accepts `reasoning_effort` / `chat_template_kwargs` extras
  (probed); they are forwarded when present in a mode's request_params.json.
- `glm-5.3-flash` is vision-capable and is the default expansion model (~2x
  faster than `qwen3.8-27b`, which is text-only).

## Conventions

- Dependency floor: fastmcp, httpx, pillow - nothing else without good reason.
- Tool docstrings are the agent-facing docs and the guaranteed-delivery
  channel (tool descriptions always reach the model; server `instructions`
  and system prompts may not). When touching workflow policy, keep it in sync
  across: generate_image/edit_image docstrings, the FastMCP `instructions`
  string, and the optional README snippet.
- Wire-contract changes come with mock assertions in `test_smoke.py` (it
  records requests in `LAST_REQ`). New tools must be added to BOTH tool-set
  assertions there (in-memory + stdio).
- To verify the effective configuration without touching files, call the
  `get_config` tool (e.g. through a live script) - env > `.env`, real env wins.

## Verification discipline

Never claim success without evidence: run the smoke test and confirm both
`ALL SMOKE TESTS PASSED` and `EXITCODE=0`. For behavior that only shows
against the real backend (sizes, seeds, expansion quality, vision), run the
matching live probe once and cite its output.
