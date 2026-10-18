"""Local service settings; no model loads at import time."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not BASE_URL.startswith("http"):
    BASE_URL = "http://" + BASE_URL
MODEL = "qwen3:0.6b"
EMBEDDING = "nomic-embed-text"
INDEX = ROOT / ".chroma"


def cool_down():
    if os.environ.get("GEX3_HARDWARE_GUARD") != "1":
        return
    def temperature():
        output = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True, timeout=10)
        value = float(output.stdout.strip())
        if value >= 80:
            raise RuntimeError("GPU reached the temperature limit")
        return value
    value = temperature()
    start = time.monotonic()
    if value >= 72:
        while value > 65:
            if time.monotonic() - start > 600:
                raise RuntimeError("GPU did not cool within ten minutes")
            time.sleep(3)
            value = temperature()


def server_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        return json.load(response)
