from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def snapshot() -> str:
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
    ) or "GPU nao detectada"
    chrome_win = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")
    chrome_win_x86 = Path(
        "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
    )
    lines = [
        "# Maquina",
        "",
        f"- os: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- distro: {_run(['lsb_release', '-ds']) or platform.version()}",
        f"- wsl: {'sim' if Path('/proc/sys/fs/binfmt_misc/WSLInterop').exists() or 'microsoft' in platform.release().lower() else 'nao'}",
        f"- python: {platform.python_version()}",
        f"- gpu: {gpu}",
        f"- which google-chrome: {shutil.which('google-chrome') or 'nao'}",
        f"- which firefox: {shutil.which('firefox') or 'nao'}",
        f"- chrome.exe: {'sim' if chrome_win.exists() or chrome_win_x86.exists() else 'nao'}",
        f"- powershell: {_run(['powershell.exe', '-NoProfile', '-Command', 'echo ok']) or 'nao'}",
        "",
        "Fatos desta maquina ficam AQUI, nao nos pesos do modelo.",
    ]
    return "\n".join(lines) + "\n"
