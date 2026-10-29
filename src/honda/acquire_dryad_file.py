"""Acquire one explicitly allow-listed small Dryad object through a real browser.

Dryad's public download route uses a JavaScript proof-of-work page. This helper
does not bypass that control: headless Chrome executes it normally. The helper
refuses unknown file IDs, unexpected sizes, or unexpected SHA-256 digests and can
never request the 103 GB data archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import websocket


ALLOWLIST = {
    "3910975": {
        "filename": "issues.zip",
        "bytes": 2_825_237,
        "sha256": "594e09131fc797af0a0cfb4355fd2dabe95d88a72fd78e73dbb3017240452a85",
    },
    "3910982": {
        "filename": "README.md",
        "bytes": 15_222,
        "sha256": "7433ae4227aa9253c3a4c3f007486f5567d7fec2fdcf33b1600799217ff7cdee",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chrome_path() -> Path:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError("Chrome/Edge not found")


def command(ws: websocket.WebSocket, message_id: int, method: str, params: dict | None = None) -> dict:
    ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
    while True:
        reply = json.loads(ws.recv())
        if reply.get("id") == message_id:
            if "error" in reply:
                raise RuntimeError(f"DevTools error: {reply['error']}")
            return reply.get("result", {})


def acquire(file_id: str, destination: Path, timeout: int = 90) -> Path:
    if file_id not in ALLOWLIST:
        raise ValueError("Dryad file ID is not allow-listed")
    expected = ALLOWLIST[file_id]
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / expected["filename"]
    if final.exists():
        if final.stat().st_size == expected["bytes"] and sha256(final) == expected["sha256"]:
            return final
        raise RuntimeError(f"Refusing to overwrite unexpected existing file: {final}")

    profile = Path(tempfile.mkdtemp(prefix="honda-h2-dryad-"))
    port = 9228
    process = subprocess.Popen(
        [
            str(chrome_path()),
            "--headless=new",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ws = None
    try:
        deadline = time.time() + timeout
        targets = None
        while time.time() < deadline:
            try:
                targets = requests.get(f"http://127.0.0.1:{port}/json", timeout=2).json()
                if targets:
                    break
            except requests.RequestException:
                time.sleep(0.25)
        if not targets:
            raise RuntimeError("Chrome DevTools endpoint did not start")
        page = next(t for t in targets if t["type"] == "page")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=10)
        command(ws, 1, "Page.enable")
        command(ws, 2, "Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(destination.resolve()), "eventsEnabled": True})
        url = f"https://datadryad.org/downloads/file_stream/{file_id}"
        command(ws, 3, "Page.navigate", {"url": url})

        partial = destination / f"{expected['filename']}.crdownload"
        while time.time() < deadline:
            if final.exists() and not partial.exists():
                break
            time.sleep(0.5)
        if not final.exists():
            state = command(
                ws,
                4,
                "Runtime.evaluate",
                {"expression": "JSON.stringify({url: location.href, title: document.title, text: document.body.innerText.slice(0, 1000)})", "returnByValue": True},
            )
            detail = state.get("result", {}).get("value", "unknown browser state")
            raise RuntimeError(f"Dryad browser download did not complete before timeout: {detail}")
    finally:
        if ws is not None:
            ws.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        shutil.rmtree(profile, ignore_errors=True)

    if final.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"Unexpected byte count for {final}: {final.stat().st_size}")
    actual = sha256(final)
    if actual != expected["sha256"]:
        raise RuntimeError(f"Unexpected SHA-256 for {final}: {actual}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_id", choices=sorted(ALLOWLIST))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(acquire(args.file_id, args.destination))


if __name__ == "__main__":
    main()
