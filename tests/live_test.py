# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=2.11", "httpx>=0.28"]
# ///
"""Live end-to-end test against the real LiteLLM server.

Uses the MCP server's own tools (same protocol path an agent uses), reviews
config without ever printing the API key, and saves results under generated/.
"""

import asyncio
import sys
from pathlib import Path

import httpx

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

GEN_PROMPT = "a red apple on a white table, studio lighting, photorealistic"
EDIT_PROMPT = "turn the apple green, keep everything else unchanged"


def paths_from(texts: list[str]) -> list[str]:
    lines = "\n".join(texts).splitlines()
    return [line.strip() for line in lines if line.strip().lower().endswith(".png")]


async def main() -> None:
    from fastmcp import Client

    import server as srv

    print(f"base_url: {srv.BASE_URL}")
    print(f"model:    {srv.MODEL}")
    print(f"api key:  {'set' if srv.API_KEY else 'MISSING'}")
    print(f"output:   {srv.OUTPUT_DIR}")

    # 0) model list - is the expected model actually registered?
    headers = {"Authorization": f"Bearer {srv.API_KEY}"} if srv.API_KEY else {}
    r = httpx.get(f"{srv.BASE_URL}/models", headers=headers, timeout=30)
    r.raise_for_status()
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    imageish = [i for i in ids if "image" in i.lower() or "minimax" in i.lower()]
    print(f"models on proxy ({len(ids)} total), image-ish: {imageish}")
    if srv.MODEL not in ids:
        print(f"WARNING: '{srv.MODEL}' not in model list - set IMAGE_MODEL in .env if needed")

    async with Client(srv.mcp) as client:
        # 1) generation
        print(f"\n[1] generate_image: {GEN_PROMPT!r}")
        res = await asyncio.wait_for(
            client.call_tool("generate_image", {"prompt": GEN_PROMPT}), timeout=620
        )
        texts = [c.text for c in res.content if c.type == "text"]
        images = [c for c in res.content if c.type == "image"]
        gen_paths = paths_from(texts)
        print(f"    -> {len(images)} inline image(s), text: {texts}")
        assert gen_paths, "no saved path in response"
        for p in gen_paths:
            f = Path(p)
            print(f"    -> file: {p} ({f.stat().st_size} bytes)")

        # 2) get_image re-serve
        print(f"\n[2] get_image: {gen_paths[0]}")
        res_g = await client.call_tool("get_image", {"path": gen_paths[0]})
        print(f"    -> blocks: {[c.type for c in res_g.content]}")

        # 3) edit pass on the generated image
        print(f"\n[3] edit_image: {EDIT_PROMPT!r} on {gen_paths[0]}")
        res2 = await asyncio.wait_for(
            client.call_tool("edit_image", {"prompt": EDIT_PROMPT, "image_paths": [gen_paths[0]]}),
            timeout=620,
        )
        texts2 = [c.text for c in res2.content if c.type == "text"]
        images2 = [c for c in res2.content if c.type == "image"]
        edit_paths = paths_from(texts2)
        print(f"    -> {len(images2)} inline image(s), text: {texts2}")
        assert edit_paths, "no saved path in edit response"
        for p in edit_paths:
            print(f"    -> file: {p} ({Path(p).stat().st_size} bytes)")

    print("\nFINAL RESULTS FOR VISION REVIEW:")
    print(f"generated: {gen_paths[0]}")
    print(f"edited:    {edit_paths[0]}")
    print("LIVE TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
