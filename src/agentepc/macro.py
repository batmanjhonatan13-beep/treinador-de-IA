from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agentepc.config import ROOT, load, resolve
from agentepc.memory import slug

HELPER = ROOT / "helpers" / "win_macro.ps1"


def macros_dir() -> Path:
    path = resolve(load()["memory"]["macros_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def macro_path(skill: str) -> Path:
    return macros_dir() / f"{slug(skill)}.json"


def to_win(path: Path) -> str:
    out = subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True)
    return out.strip()


def ping() -> str:
    proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            to_win(HELPER),
            "-Mode",
            "ping",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    text = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(text or f"win_macro ping falhou ({proc.returncode})")
    return text


def _ps(mode: str, extra: list[str], window: bool) -> int:
    args = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        to_win(HELPER),
        "-Mode",
        mode,
        *extra,
    ]
    if window:
        cmd = ["cmd.exe", "/c", "start", "/wait", "agente-pc", *args]
        return subprocess.run(cmd, check=False).returncode
    return subprocess.run(args, check=False).returncode


def record(skill: str, window: bool = True) -> Path:
    dest = macro_path(skill)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    code = _ps(
        "record",
        ["-OutFile", to_win(dest), "-Skill", skill],
        window=window,
    )
    if code == 2:
        raise RuntimeError("captura cancelada (F10). nada salvo.")
    if code != 0 or not dest.exists():
        raise RuntimeError("captura falhou ou ficou vazia.")
    return dest


def replay(skill: str, window: bool = False) -> None:
    dest = macro_path(skill)
    if not dest.exists():
        raise FileNotFoundError(f"macro inexistente: {dest}")
    code = _ps("replay", ["-InFile", to_win(dest)], window=window)
    if code == 2:
        raise RuntimeError("replay cancelado (F10).")
    if code != 0:
        raise RuntimeError("replay falhou.")


def load_macro(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def summarize(events: list[dict], limit: int = 40) -> str:
    lines: list[str] = []
    for ev in events[:limit]:
        kind = ev.get("type")
        if kind == "mouse_move":
            continue
        if kind in {"mouse_down", "mouse_up"}:
            lines.append(f"- {kind} {ev.get('button')} @ {ev.get('x')},{ev.get('y')}")
        elif kind in {"key_down", "key_up"}:
            lines.append(f"- {kind} {ev.get('name') or ev.get('vk')}")
        else:
            lines.append(f"- {kind}")
    skipped_moves = sum(1 for ev in events if ev.get("type") == "mouse_move")
    extra = max(0, len(events) - limit)
    if skipped_moves:
        lines.append(f"- ({skipped_moves} movimentos de mouse compactados)")
    if extra:
        lines.append(f"- ... +{extra} eventos")
    return "\n".join(lines) if lines else "- (sem eventos)"


def procedure_body(skill: str, task: str, macro: dict) -> str:
    events = macro.get("events") or []
    screen = macro.get("screen") or {}
    return (
        f"# {skill}\n\n"
        f"Tarefa: {task}\n\n"
        f"Tipo: macro teclado/mouse (windows)\n"
        f"Arquivo: data/macros/{slug(skill)}.json\n"
        f"Tela: {screen.get('w')}x{screen.get('h')}\n"
        f"Eventos: {len(events)}\n\n"
        f"Replay:\n```\npython -m agentepc replay --skill {slug(skill)}\n```\n\n"
        f"Passos:\n{summarize(events)}\n"
    )
