from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

# Torch 2.14 tenta compilar Triton no 1o passo; no WSL sem gcc-dev isso quebra.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from agentepc import bake
from agentepc.config import load, resolve


def adapter_dir() -> Path:
    return resolve(load()["promote"]["adapter_dir"])


def adapter_exists() -> bool:
    return (adapter_dir() / "adapter_config.json").exists()


_stop = threading.Event()


def efficiency() -> dict:
    """Rodadas de treino por linha, medidas nos ultimos treinos que fecharam 100%."""
    done = [r for r in _rows() if r["state"] == "done" and r.get("train_rounds") and r.get("facts")]
    recent = done[-5:]
    if not recent:
        return {"ratio": None, "samples": 0}
    ratio = sum(r["train_rounds"] / r["facts"] for r in recent) / len(recent)
    return {"ratio": round(ratio, 3), "samples": len(recent)}


def predict_rounds(n_facts: int) -> int | None:
    """Ex.: 20 linhas fecharam com 27 rodadas -> 40 linhas: 54. Sem historico, None."""
    ratio = efficiency()["ratio"]
    return max(1, math.ceil(ratio * n_facts)) if ratio else None


def plan(force: bool = False, file: str | None = None) -> dict:
    ok, reason, stats = bake.ready(file)
    stats["lora"] = adapter_exists()
    if not ok and not force:
        return {
            "will_train": False,
            "reason": reason,
            "stats": stats,
            "file_always": True,
            "next": "escreva mais em data/sobre-mim.md ou use --force",
        }
    dataset = bake.to_dataset(file)
    cfg = load()
    return {
        "will_train": True,
        "reason": reason if ok else "force: lote pequeno, risco de viciar",
        "stats": {**stats, "examples": _count(dataset)},
        "predicted_rounds": predict_rounds(stats["facts"]),
        "efficiency": efficiency(),
        "dataset": str(dataset),
        "base": cfg["model"]["base_hf"],
        "adapter_dir": str(adapter_dir()),
        "method": "QLoRA",
        "file_always": True,
        "note": "Depois do LoRA o chat continua lendo o arquivo. Os dois.",
        "install": "uv venv .venv-train && uv pip install -r requirements-train.txt",
    }


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run(force: bool = False, file: str | None = None, rounds: int = 1) -> dict:
    """Uma passada avulsa (linha de comando). A pagina usa o ciclo em start_job."""
    from agentepc.session import Missing, TrainSession

    _stop.clear()
    info = plan(force=force, file=file)
    if not info["will_train"]:
        return info
    try:
        s = TrainSession(bake.training_pairs(file), _stop)
        s.open(resume_from=adapter_dir() if adapter_exists() else None)
    except Missing as exc:
        return {**info, "ran": False, "error": str(exc)}
    try:
        if not s.train(rounds):
            return {**info, "ran": False, "stopped": True}
        s.save(adapter_dir())
    finally:
        s.close()
    return {**info, "ran": True, "adapter": str(adapter_dir()), "rounds": rounds}


_job: dict = {
    "state": "idle",
    "id": "",
    "log": "",
    "lines": [],
    "saved": False,
    "result": None,
    "file": "",
    "round": 0,
    "passed": 0,
    "total": 0,
    "eval": [],
    "rounds": [],
    "stopping": False,
    "planned": None,
    "train_rounds": 0,
    "streak": 0,
    "need_streak": 3,
    "attempts": [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _history_path() -> Path:
    path = resolve(load()["learn"]["history_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _logs_dir() -> Path:
    path = resolve("data/logs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _versions_dir() -> Path:
    return adapter_dir().parent / "versions"


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,40}", value or ""):
        raise ValueError("id invalido")
    return value


def _rows() -> list[dict]:
    path = _history_path()
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row.setdefault("id", re.sub(r"\D", "", row.get("ts", ""))[:14])
        row.setdefault("snapshot", False)
        row.setdefault("parent", None)
    return rows


def _write_rows(rows: list[dict]) -> None:
    _history_path().write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def _copy_adapter(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(adapter_dir(), dst, ignore=shutil.ignore_patterns("runs"))


def _ensure_legacy() -> None:
    """LoRA que ja existia antes das versoes vira a versao 'legado', para poder ser apagada por botao."""
    if _job["state"] == "running" or not adapter_exists():
        return
    rows = _rows()
    if any(r["snapshot"] for r in rows):
        return
    _copy_adapter(_versions_dir() / "legado")
    rows.append(
        {
            "id": "legado",
            "ts": _now(),
            "file": "(treinos anteriores ao controle de versao)",
            "state": "legado",
            "rounds": 0,
            "passed": 0,
            "total": 0,
            "per_round": [],
            "parent": None,
            "snapshot": True,
        }
    )
    _write_rows(rows)


def history(limit: int = 30) -> list[dict]:
    _travadas.clear()
    _ensure_legacy()
    rows = _rows()
    for row in rows:
        snap = _versions_dir() / row["id"]
        row["size"] = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file()) if row["snapshot"] and snap.exists() else 0
    return rows[-limit:]


def _say(msg: str) -> None:
    """Uma linha de log: vai pro estado atual e para o arquivo daquele treino."""
    _job["log"] = msg
    line = {"ts": _now(), "msg": msg}
    _job["lines"].append(line)
    if _job["id"]:
        with (_logs_dir() / f"{_job['id']}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_log(run_id: str = "") -> dict:
    """Log so daquele treino. Sem id, devolve o treino atual/ultimo."""
    if not run_id or run_id == _job["id"]:
        return {"id": _job["id"], "running": _job["state"] == "running", "lines": list(_job["lines"])}
    path = _logs_dir() / f"{_safe_id(run_id)}.jsonl"
    lines = []
    if path.exists():
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"id": run_id, "running": False, "lines": lines}


def _record(state: str, snapshot: bool, parent: str | None) -> None:
    rows = _rows()
    try:
        facts = bake.facts_from_knowledge(_job["file"])
    except ValueError:
        facts = []
    rows.append(
        {
            "id": _job["id"],
            "ts": _now(),
            "file": _job["file"],
            "state": state,
            "rounds": _job["round"],
            "passed": _job["rounds"][-1]["passed"] if _job["rounds"] else 0,
            "total": _job["total"],
            "per_round": _job["rounds"],
            "parent": parent,
            "snapshot": snapshot,
            "facts": len(facts),
            "sample": [f[:90] for f in facts[:3]],
            # o adaptador so serve no modelo em que foi treinado; sem isso ele vira lixo
            "base_hf": (load().get("model") or {}).get("base_hf"),
            "modelo": (load().get("model") or {}).get("name"),
            "train_rounds": _job["train_rounds"],
            "planned": _job["planned"],
            # consolidado = N tentativas SEGUIDAS, cada uma do zero, acertando tudo
            # ja na primeira prova. Uma tentativa isolada pode ser sorte de rodada.
            "first_pass": bool(_job["attempts"] and _job["attempts"][0].get("first")),
            "streak": _job["streak"],
            "attempts": _job["attempts"],
            "consolidated": _job["streak"] >= _job["need_streak"],
        }
    )
    _write_rows(rows)


def delete_training(run_id: str) -> dict:  # usado so pela linha de comando
    """Apaga os pesos daquele treino. Modelos ja fundidos nao sao afetados."""
    if _job["state"] == "running":
        return {"ok": False, "reason": "treino em andamento; pare antes de apagar"}
    _safe_id(run_id)
    rows = _rows()
    target = next((r for r in rows if r["id"] == run_id), None)
    if not target or not target["snapshot"]:
        return {"ok": False, "reason": "esse treino nao tem pesos guardados"}
    doomed = {run_id}  # treinos sao independentes: so este sai
    removed = []
    for r in rows:
        if r["id"] in doomed and r["snapshot"]:
            shutil.rmtree(_versions_dir() / r["id"], ignore_errors=True)
            r["snapshot"] = False
            r["deleted"] = True
            removed.append(r["id"])
    if _active_raw() in doomed:
        set_active("")
    left = [r for r in rows if r["snapshot"]]
    if adapter_dir().exists():
        shutil.rmtree(adapter_dir())
    if left:
        shutil.copytree(_versions_dir() / left[-1]["id"], adapter_dir())
    _write_rows(rows)
    try:
        from agentepc import engine

        engine.unload()
    except Exception:
        pass
    return {"ok": True, "removed": removed, "active": left[-1]["id"] if left else None}


def _active_file() -> Path:
    return resolve("data/active_adapter.txt")


def _active_raw() -> str:
    try:
        return _active_file().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def set_active(value: str) -> None:
    """"" = ultimo treino (padrao), "none" = so o Qwen original, ou o id de um treino."""
    if value not in ("", "none"):
        if not (_versions_dir() / _safe_id(value)).exists():
            raise ValueError("treino sem pesos guardados")
    _active_file().write_text(value, encoding="utf-8")
    try:
        from agentepc import engine

        engine.unload()
    except Exception:
        pass


def base_do_treino(run_id: str) -> str | None:
    return next((r.get("base_hf") for r in _rows() if r["id"] == run_id), None)


def combina_com_modelo(run_id: str) -> bool:
    """Treino feito em outro modelo base nao serve no modelo atual."""
    base = base_do_treino(run_id)
    return not base or base == (load().get("model") or {}).get("base_hf")


def serving_adapter() -> Path | None:
    """Adaptador que o chat aplica no modelo atual. Um por vez, e so se for do mesmo base."""
    raw = _active_raw()
    if raw == "none":
        return None
    if raw and (_versions_dir() / raw).exists():
        return _versions_dir() / raw if combina_com_modelo(raw) else None
    return adapter_dir() if adapter_exists() else None


# Juntar dois conhecimentos JA treinados nao funciona. Medido duas vezes:
#   - somar um adaptador sobre o outro: respostas viram ruido (0/12);
#   - concatenar as matrizes (o "cat" do PEFT): um conhecimento responde pelo outro
#     ("mae da Carla?" -> "Maura", que e a mae do Jhonatan) — 3/11.
# Para ter dois assuntos no mesmo peso, treine os dois arquivos JUNTOS: no Treino
# selecione mais de um arquivo e o ciclo usa "a.md + b.md" como um so conhecimento.


def consolidated_files() -> dict:
    """Arquivo -> treino que fechou 100%. Serve para travar o retreino e liberar o export."""
    out = {}
    for r in _rows():
        if r.get("consolidated") and r.get("snapshot") and r.get("file"):
            out[r["file"]] = r["id"]
    return out


def adapters() -> dict:
    _travadas.clear()
    _ensure_legacy()
    raw = _active_raw()
    snaps = [r for r in history(100) if r["snapshot"]]
    if raw == "none" or not snaps:
        active = "none"
    elif raw and any(r["id"] == raw for r in snaps):
        active = raw
    else:
        active = snaps[-1]["id"]  # padrao: o treino mais recente
    keys = ("id", "ts", "file", "state", "passed", "total", "facts", "size", "consolidated",
            "first_pass", "tested", "base_hf", "modelo")
    base_atual = (load().get("model") or {}).get("base_hf")
    itens = []
    for r in snaps:
        item = {k: r.get(k) for k in keys}
        item["serve_no_modelo"] = not r.get("base_hf") or r.get("base_hf") == base_atual
        itens.append(item)
    return {"active": active, "items": itens, "base_atual": base_atual}


_tjob: dict = {"state": "idle", "id": "", "text": ""}


def test_status() -> dict:
    return dict(_tjob)


def test_group(file: str) -> dict:
    """Prova TODOS os treinos daquele arquivo, um de cada vez, no mesmo modelo carregado.

    Cada treino e uma tentativa completa a partir do Qwen original — eles nao se somam.
    Testar o grupo mostra qual tentativa realmente fixou o conhecimento.
    """
    if _job["state"] == "running" or _tjob["state"] == "running":
        return {"accepted": False, "reason": "treino ou teste em andamento"}
    alvos = [r for r in _rows() if r.get("file") == file and r.get("snapshot")]
    if not alvos:
        return {"accepted": False, "reason": "esse arquivo nao tem pesos guardados"}
    if not file or file.startswith("("):
        return {"accepted": False, "reason": "treino antigo sem arquivo de origem: nao ha perguntas para fazer"}
    _tjob.update({"state": "running", "id": file, "text": f"{len(alvos)} treino(s) para provar"})

    def _go() -> None:
        session = None
        try:
            from agentepc import engine, exam, ollama
            from agentepc.session import TrainSession

            quiz = exam.build(file)
            try:
                ollama.stop_model()
            except Exception:
                pass
            engine.unload()
            _tjob["text"] = "carregando o modelo uma vez para provar todos"
            # evento proprio: um "parar" de treino antigo nao pode cortar este teste
            session = TrainSession([], threading.Event(), lambda m: _tjob.update({"text": m}))
            session.open(resume_from=_versions_dir() / alvos[0]["id"])
            notas: dict[str, list[int]] = {}
            for i, row in enumerate(alvos):
                nome = "default" if i == 0 else f"v{i}"
                if i:
                    session.add_adapter(_versions_dir() / row["id"], nome)
                session.use_adapter(nome)
                _tjob["text"] = f"provando treino {i + 1}/{len(alvos)} ({row['id']})"
                answers = session.answer([q["q"] for q in quiz])
                ok = sum(exam.grade(a, item)["ok"] for a, item in zip(answers, quiz))
                notas[row["id"]] = [ok, len(quiz)]
            fresh = _rows()
            for r in fresh:
                if r["id"] in notas:
                    r["tested"] = notas[r["id"]]
                    # o verde vem da sequencia no treino; o teste so derruba quem nao aguenta
                    if r.get("consolidated") and notas[r["id"]][0] < len(quiz):
                        r["consolidated"] = False
            _write_rows(fresh)
            melhor = max(notas.values(), key=lambda v: v[0])
            fechou = sum(1 for v in notas.values() if v[0] == len(quiz))
            _tjob.update({
                "state": "done",
                "text": f"{len(alvos)} treino(s) provados: melhor {melhor[0]}/{melhor[1]}"
                + (f", {fechou} consolidado(s)" if fechou else " — nenhum consolidado ainda"),
            })
        except Exception as exc:
            _tjob.update({"state": "error", "text": str(exc)})
        finally:
            if session is not None:
                session.close()

    threading.Thread(target=_go, name="consolidation-test", daemon=True).start()
    return {"accepted": True}


def delete_group(file: str) -> dict:
    """Apaga tudo daquele arquivo: pesos, logs e as linhas do historico.

    Pacotes ja exportados nao sao afetados — eles tem a copia dos pesos dentro.
    """
    if _job["state"] == "running":
        return {"ok": False, "reason": "treino em andamento; pare antes de apagar"}
    rows = _rows()
    alvos = [r for r in rows if r.get("file") == file]
    if not alvos:
        return {"ok": False, "reason": "nao ha treino desse arquivo"}
    removidos = []
    for r in [x for x in rows if x.get("file") == file]:
        shutil.rmtree(_versions_dir() / r["id"], ignore_errors=True)
        (_logs_dir() / f"{r['id']}.jsonl").unlink(missing_ok=True)
        removidos.append(r["id"])
    rows = [r for r in rows if r.get("file") != file]
    if _active_raw() in removidos:
        set_active("")
    restantes = [r for r in rows if r["snapshot"]]
    if adapter_dir().exists():
        shutil.rmtree(adapter_dir())
    if restantes:
        shutil.copytree(_versions_dir() / restantes[-1]["id"], adapter_dir())
    _write_rows(rows)
    try:
        from agentepc import engine

        engine.unload()
    except Exception:
        pass
    return {"ok": True, "removed": removidos, "active": restantes[-1]["id"] if restantes else None}


def job_status() -> dict:
    return {
        "state": _job["state"],
        "id": _job["id"],
        "log": _job["log"],
        "result": _job["result"],
        "lora": adapter_exists(),
        "file": _job["file"],
        "round": _job["round"],
        "passed": _job["passed"],
        "total": _job["total"],
        "eval": _job["eval"],
        "rounds": _job["rounds"],
        "stopping": _job["stopping"],
        "planned": _job["planned"],
        "train_rounds": _job["train_rounds"],
        "streak": _job["streak"],
        "need_streak": _job["need_streak"],
        "attempts": _job["attempts"],
        "efficiency": efficiency(),
        "consolidated_files": consolidated_files(),
        "test": test_status(),
        "spinning": _job["state"] == "running",
        "history": history(),
    }


# Uma pergunta que a prova nunca aceita prende o ciclo para sempre — com teto zero, que e
# o padrao, o treino roda a noite inteira gastando GPU a toa. Aqui se conta quantas provas
# seguidas a MESMA pergunta erra com a MESMA resposta: se passar do limite, o ciclo para e
# mostra o caso, em vez de insistir.
_TRAVA_LIMITE = 12
_travadas: dict[str, list] = {}


def _marca_travada(rows: list[dict]) -> dict | None:
    for row in rows:
        chave = row["q"]
        if row["ok"]:
            _travadas.pop(chave, None)
            continue
        resposta = " ".join((row.get("answer") or "").lower().split())[:200]
        antes = _travadas.get(chave)
        if antes and antes[0] == resposta:
            antes[1] += 1
        else:
            _travadas[chave] = [resposta, 1]
        if _travadas[chave][1] >= _TRAVA_LIMITE:
            return {**row, "vezes": _travadas[chave][1]}
    return None


def _probe(session, quiz: list[dict]) -> list[dict] | None:
    """Pergunta tudo com o adaptador e sem caderno, no modelo que ja esta na GPU."""
    from agentepc import exam

    answers = session.answer([item["q"] for item in quiz])
    if _stop.is_set() or len(answers) < len(quiz):
        return None
    rows = [exam.grade(a, item) for a, item in zip(answers, quiz)]
    _job["eval"] = rows
    _job["passed"] = sum(1 for r in rows if r["ok"])
    _job["total"] = len(rows)
    for row in rows:
        _say(f"{'CERTO' if row['ok'] else 'ERRADO'}: {row['q']} -> {row['answer'][:120]}")
    return rows


def _aviso_travada(row: dict) -> str:
    esperado = ", ".join(row.get("keys") or []) or "(sem palavras-chave)"
    return (
        f"PAREI: a pergunta \"{row['q']}\" foi respondida {row['vezes']} vezes seguidas com "
        f"\"{(row.get('answer') or '')[:80]}\", e a prova exige as palavras: {esperado}.\n"
        "Se a resposta quer dizer a mesma coisa, quem esta apertada e a prova: reescreva o "
        "fato no arquivo do jeito que voce quer ouvir a resposta e treine de novo.\n"
        "Se nao quer dizer, o modelo nao aprendeu esse fato — costuma ser fato longo demais "
        "ou dois fatos na mesma linha; separe em dois e treine de novo.\n"
        "Continuar insistindo so gastaria GPU: o ciclo nao tem teto de rodadas."
    )


def stop_job() -> dict:
    """Interrompe o ciclo. Seguro: o que ja foi salvo continua, a rodada em curso e descartada."""
    if _job["state"] != "running":
        return {**job_status(), "accepted": False, "reason": "nao ha treino em andamento"}
    _stop.set()
    _job["stopping"] = True
    _say("parando: termina o passo atual e descarta a rodada em curso (o LoRA salvo fica intacto)")
    return {**job_status(), "accepted": True, "reason": "parada pedida"}


def start_job(force: bool = False, file: str | None = None, fresh: bool = False) -> dict:
    if _job["state"] == "running":
        return {**job_status(), "accepted": False, "reason": "treino ja em andamento"}
    if not file:
        return {**job_status(), "accepted": False, "state": "idle", "reason": "escolha UM arquivo para treinar"}
    ja = consolidated_files().get(file)
    if ja and not force:
        return {
            **job_status(),
            "accepted": False,
            "state": "idle",
            "reason": f"{file} ja esta consolidado (treino {ja}): o modelo acerta tudo sem consulta. "
            "Para treinar de novo, apague esse conhecimento na lista abaixo.",
        }
    try:
        preview = plan(force=force, file=file)
    except ValueError as exc:
        return {**job_status(), "accepted": False, "state": "idle", "reason": str(exc)}
    if not preview.get("will_train"):
        return {**preview, "accepted": False, "state": "idle"}

    _travadas.clear()
    _ensure_legacy()
    # Cada treino nasce do Qwen original. Nao ha heranca entre treinos: quem junta conhecimentos e a fusao.
    parent = None
    _stop.clear()
    _job.update(
        {
            "state": "running",
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "lines": [],
            "saved": False,
            "result": None,
            "file": file,
            "round": 0,
            "passed": 0,
            "total": 0,
            "eval": [],
            "rounds": [],
            "stopping": False,
            "planned": None,
            "train_rounds": 0,
            "streak": 0,
            "need_streak": int((load().get("learn") or {}).get("streak") or 3),
            "attempts": [],
        }
    )
    _say(f"treino de {file} (independente, parte do Qwen original): elaborando perguntas a partir do arquivo")
    max_rounds = int((load().get("learn") or {}).get("max_rounds") or 0)

    def _finish(state: str, log: str, result: dict | None) -> None:
        snap = bool(_job["saved"])
        if snap:
            _copy_adapter(_versions_dir() / _job["id"])
        _say(log + (" Pesos desta versao guardados." if snap else ""))
        _job["state"] = state
        _job["result"] = result
        _job["stopping"] = False
        _record(state, snap, parent)

    def _go() -> None:
        session = None
        try:
            from agentepc import engine, exam, ollama
            from agentepc.session import TrainSession

            quiz = exam.build(file)
            if not quiz:
                _finish("error", "nao consegui gerar perguntas desse arquivo (precisa de fatos em linhas '- ...').", None)
                return
            _job["total"] = len(quiz)
            novas = sum(1 for q in quiz if q.get("kind") == "frase")
            _say(f"{len(quiz)} perguntas na prova ({novas} redigidas pelo modelo, o resto completa-frase)")

            pairs = bake.training_pairs(file)
            n_facts = len(bake.facts_from_knowledge(file))
            planned = predict_rounds(n_facts)
            chunk = max(1, int((load().get("learn") or {}).get("chunk_rounds") or 6))
            _job["planned"] = planned
            eff = efficiency()
            if planned:
                _say(
                    f"historico: {eff['ratio']} rodadas por linha ({eff['samples']} treino(s)); "
                    f"{n_facts} linhas = {planned} rodadas seguidas antes da primeira prova"
                )
            else:
                _say("sem treino concluido no historico: prova a cada rodada ate calibrar")

            try:
                ollama.stop_model()
            except Exception:
                pass
            engine.unload()
            session = TrainSession(pairs, _stop, _say)
            session.open(resume_from=None)  # cada treino parte do Qwen original
            _say(f"{len(pairs)} exemplos de treino; o modelo fica na GPU ate o fim do ciclo")

            alvo = _job["need_streak"]
            planejado = planned or 1
            tentativa = 0
            while True:
                tentativa += 1
                _job["round"] = tentativa
                _job["eval"] = []
                _job["passed"] = 0
                # cada tentativa nasce do Qwen original: e isso que torna a sequencia uma prova
                if session is not None:
                    session.close()
                session = TrainSession(pairs, _stop, _say)
                session.open(resume_from=None)
                _say(f"tentativa {tentativa} (sequencia {_job['streak']}/{alvo}): {planejado} rodadas do zero")
                feitas = 0
                while feitas < planejado:
                    n = min(chunk, planejado - feitas)
                    if not session.train(n):
                        _finish("stopped", f"interrompido na tentativa {tentativa}.", None)
                        return
                    feitas += n
                    _job["train_rounds"] += n
                rows = _probe(session, quiz)
                if rows is None:
                    _finish("stopped", f"interrompido na prova da tentativa {tentativa}.", None)
                    return
                travada = _marca_travada(rows)
                if travada:
                    _finish("travado", _aviso_travada(travada), None)
                    return
                limpa = all(r["ok"] for r in rows)
                _job["rounds"].append(
                    {"round": tentativa, "train_rounds": _job["train_rounds"], "passed": _job["passed"], "total": len(quiz)}
                )
                if limpa:
                    _job["streak"] += 1
                    _job["attempts"].append({"rounds": planejado, "passed": _job["passed"], "total": len(quiz), "first": True})
                    session.save(adapter_dir())
                    _job["saved"] = True
                    _say(f"tentativa {tentativa}: {len(quiz)}/{len(quiz)} de primeira — sequencia {_job['streak']}/{alvo}")
                    if _job["streak"] >= alvo:
                        _finish(
                            "done",
                            f"consolidado: {alvo} tentativas seguidas acertaram tudo de primeira, "
                            f"com {planejado} rodadas cada ({planejado / max(n_facts, 1):.2f} por linha).",
                            None,
                        )
                        return
                    continue
                # falhou de primeira: a sequencia zera e o ciclo descobre quantas rodadas faltam
                _job["streak"] = 0
                _say(f"tentativa {tentativa}: {_job['passed']}/{len(quiz)} — sequencia zerada; treinando ate fechar para calibrar")
                extra = 0
                while True:
                    if not session.train(1):
                        _finish("stopped", f"interrompido na tentativa {tentativa}.", None)
                        return
                    extra += 1
                    _job["train_rounds"] += 1
                    rows = _probe(session, quiz)
                    if rows is None:
                        _finish("stopped", f"interrompido na prova da tentativa {tentativa}.", None)
                        return
                    travada = _marca_travada(rows)
                    if travada:
                        _finish("travado", _aviso_travada(travada), None)
                        return
                    _job["rounds"].append(
                        {"round": tentativa, "train_rounds": _job["train_rounds"], "passed": _job["passed"], "total": len(quiz)}
                    )
                    if all(r["ok"] for r in rows):
                        break
                    if max_rounds and extra >= max_rounds:
                        _finish("error", f"parou no teto com {_job['passed']}/{len(quiz)}.", None)
                        return
                session.save(adapter_dir())
                _job["saved"] = True
                _job["attempts"].append(
                    {"rounds": planejado + extra, "passed": len(quiz), "total": len(quiz), "first": False}
                )
                planejado += extra
                _say(f"tentativa {tentativa} fechou com {planejado} rodadas; a proxima ja comeca com esse numero")
        except Exception as exc:
            _finish("error", str(exc), {"error": str(exc)})
        finally:
            if session is not None:
                session.close()

    threading.Thread(target=_go, name="qlora-loop", daemon=True).start()
    return {**job_status(), "accepted": True, "reason": "ciclo de treino+prova iniciado"}
