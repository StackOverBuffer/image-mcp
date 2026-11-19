# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28", "pillow>=10.0"]
# ///
"""Probe whether the image backend honors a `seed` parameter.

Sends the same prompt/size with seed=42 twice (byte-identical output => seed is
honored and generation is deterministic) and once with seed=43 (must differ,
else something is caching). Uses 2048x2048 to keep it quick.
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import httpx

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

import server as srv  # noqa: E402  (loads .env / config)

PROMPT = "a red apple on a white table, studio lighting, photorealistic"


def generate(seed: int | None) -> bytes:
    payload = {"model": srv.MODEL, "prompt": PROMPT, "size": "2048x2048"}
    if seed is not None:
        payload["seed"] = seed
    r = httpx.post(
        f"{srv.BASE_URL}/images/generations",
        headers={"Authorization": f"Bearer {srv.API_KEY}"} if srv.API_KEY else {},
        json=payload,
        timeout=700,
    )
    r.raise_for_status()
    data = r.json()["data"][0]
    if data.get("b64_json"):
        return base64.b64decode(data["b64_json"])
    return httpx.get(data["url"], timeout=120).content


def main() -> None:
    h1 = hashlib.sha256(generate(42)).hexdigest()
    h2 = hashlib.sha256(generate(42)).hexdigest()
    h3 = hashlib.sha256(generate(43)).hexdigest()
    print(f"seed 42 run 1: {h1[:16]}")
    print(f"seed 42 run 2: {h2[:16]}")
    print(f"seed 43 run 1: {h3[:16]}")
    if h1 == h2 and h1 != h3:
        print("RESULT: seed HONORED - same seed reproduces the same image")
    elif h1 == h2 == h3:
        print("RESULT: suspicious - all identical even across seeds (caching?)")
    else:
        print("RESULT: seed IGNORED (or sampler nondeterministic) - do not add a seed param")


if __name__ == "__main__":
    main()
