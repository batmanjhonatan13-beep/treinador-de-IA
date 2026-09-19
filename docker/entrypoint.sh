#!/bin/sh
set -eu

python3 - <<'PY'
import json
import os
import time
import urllib.request

host = os.environ.get("OLLAMA_HOST", "ollama:11434")
host = host.replace("http://", "").replace("https://", "")
url = f"http://{host}"
print(f"esperando ollama em {url} ...", flush=True)
ok = False
for _ in range(90):
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=2).read()
        ok = True
        break
    except Exception:
        time.sleep(1)
if not ok:
    raise SystemExit("ollama nao subiu")

model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
tags = urllib.request.urlopen(f"{url}/api/tags", timeout=5).read().decode()
if model.split(":")[0] not in tags:
    print(f"baixando {model} (primeira vez) ...", flush=True)
    req = urllib.request.Request(
        f"{url}/api/pull",
        data=json.dumps({"name": model, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=600)
print("chat em 0.0.0.0:8765", flush=True)
PY

exec python -m agentepc web --host 0.0.0.0 --port 8765
