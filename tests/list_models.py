# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""List models available on the configured LiteLLM proxy (reads .env via server.py)."""

import httpx

from pathlib import Path
import sys

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

import server as srv  # noqa: F401,E402  (loads .env / config on import)


def main() -> None:
    headers = {"Authorization": f"Bearer {srv.API_KEY}"} if srv.API_KEY else {}
    r = httpx.get(f"{srv.BASE_URL}/models", headers=headers, timeout=30)
    r.raise_for_status()
    for m in r.json().get("data", []):
        print(m.get("id", ""))


if __name__ == "__main__":
    main()
