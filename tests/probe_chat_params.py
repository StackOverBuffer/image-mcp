# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Probe whether the proxy/backend accepts non-standard chat parameters
(reasoning_effort, chat_template_kwargs) or rejects them.
"""

import sys
from pathlib import Path

import httpx

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

import server as srv  # noqa: E402


def try_body(label: str, extra: dict) -> None:
    body = {
        "model": srv.PROMPT_LLM_MODEL or srv.MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 30,
        **extra,
    }
    try:
        r = httpx.post(
            f"{srv.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {srv.API_KEY}"} if srv.API_KEY else {},
            json=body,
            timeout=120,
        )
        detail = r.text[:160].replace("\n", " ")
        print(f"{label}: HTTP {r.status_code}  {detail}")
    except Exception as exc:
        print(f"{label}: ERROR {str(exc)[:160]}")


def main() -> None:
    try_body("baseline (no extras)", {})
    try_body("reasoning_effort=low", {"reasoning_effort": "low"})
    try_body("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}})
    try_body("both", {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}})


if __name__ == "__main__":
    main()
