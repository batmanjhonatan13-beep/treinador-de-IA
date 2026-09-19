from __future__ import annotations

import argparse
import json

import time
from pathlib import Path

from agentepc import bake, episodes, macro, memory, probe, promote, train, web


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caderno na hora; peso so em lote. Um sucesso nao e treino."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="inspeciona a maquina e grava data/machine.md")

    rec = sub.add_parser("record", help="grava um episodio (sucesso ou erro)")
    rec.add_argument("--skill", required=True)
    rec.add_argument("--task", required=True)
    rec.add_argument("--command", required=True)
    rec.add_argument("--ok", action="store_true")
    rec.add_argument("--notes", default="")

    sub.add_parser("status", help="mostra se o lote ja pode virar peso")
    sub.add_parser("dataset", help="gera data/datasets/sft.jsonl sem treinar")

    tr = sub.add_parser("train-plan", help="mostra o plano de LoRA; nao baixa modelo")
    tr.add_argument("--force", action="store_true")

    cap = sub.add_parser(
        "capture",
        help="grava teclado/mouse no Windows e vira skill no caderno",
    )
    cap.add_argument("--skill", required=True)
    cap.add_argument("--task", default="")
    cap.add_argument("--notes", default="")
    cap.add_argument("--import-json", dest="import_json", default="")
    cap.add_argument("--no-window", action="store_true")

    rep = sub.add_parser("replay", help="repete uma macro gravada")
    rep.add_argument("--skill", required=True)
    rep.add_argument("--window", action="store_true")

    chat = sub.add_parser("web", help="abre o chat local no navegador (127.0.0.1)")
    chat.add_argument("--host", default="0.0.0.0")
    chat.add_argument("--port", type=int, default=8765)

    teach = sub.add_parser("teach", help="grava um fato sobre voce no caderno (nao no peso)")
    teach.add_argument("--text", required=True)

    learn = sub.add_parser("learn", help="arquivo -> dataset; com --train tenta QLoRA")
    learn.add_argument("--train", action="store_true")
    learn.add_argument("--force", action="store_true")

    imp = sub.add_parser("import-file", help="copia um .md/.txt/.jsonl para data/conhecimento/")
    imp.add_argument("--file", required=True)
    imp.add_argument("--name", default="")

    args = parser.parse_args()

    if args.cmd == "probe":
        text = probe.snapshot()
        path = memory.write_machine(text)
        print(text)
        print(f"gravado em {path}")
        return

    if args.cmd == "record":
        row = episodes.append(
            skill=args.skill,
            task=args.task,
            command=args.command,
            ok=args.ok,
            notes=args.notes,
            source="human",
        )
        if args.ok:
            proc = memory.upsert_procedure(
                args.skill,
                (
                    f"# {args.skill}\n\n"
                    f"Tarefa: {args.task}\n\n"
                    f"```\n{args.command}\n```\n\n"
                    f"{args.notes}"
                ),
            )
            print("sucesso -> caderno (consulta). pesos intactos.")
            print(f"procedimento: {proc}")
        else:
            print("erro registrado. nada foi promovido.")
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return

    if args.cmd == "status":
        ok, reason, stats = promote.ready()
        print(json.dumps({"ready": ok, "reason": reason, **stats}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "dataset":
        path = bake.to_dataset()
        print(f"dataset: {path}")
        return

    if args.cmd == "train-plan":
        print(json.dumps(train.plan(force=args.force), ensure_ascii=False, indent=2))
        return

    if args.cmd == "capture":
        task = args.task or args.skill
        print(
            "Sessao explicita. F8 salva, F9 pausa (senha), F10 cancela.\n"
            "Nao digite senha/cartao enquanto GRAVANDO. Isso vai pro caderno, nao pro peso."
        )
        if args.import_json:
            src = Path(args.import_json)
            dest = macro.macro_path(args.skill)
            dest.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")
            path = dest
        else:
            path = macro.record(args.skill, window=not args.no_window)
        data = macro.load_macro(path)
        events = data.get("events") or []
        if not events:
            raise SystemExit("macro vazia. nada promovido.")
        body = macro.procedure_body(args.skill, task, data)
        if args.notes:
            body += f"\nNotas: {args.notes}\n"
        proc = memory.upsert_procedure(args.skill, body)
        episodes.append(
            skill=args.skill,
            task=task,
            command=f"python -m agentepc replay --skill {memory.slug(args.skill)}",
            ok=True,
            notes=args.notes or f"macro {len(events)} eventos",
            source="macro",
        )
        print(f"macro: {path}")
        print(f"procedimento: {proc}")
        print(f"{len(events)} eventos -> caderno. pesos intactos.")
        return

    if args.cmd == "replay":
        print("replay em 2s. F10 cancela. mouse e teclado vao se mover.")
        time.sleep(2)
        macro.replay(args.skill, window=args.window)
        print("replay ok")
        return

    if args.cmd == "web":
        web.serve(host=args.host, port=args.port)
        return

    if args.cmd == "teach":
        path = memory.append_profile(args.text)
        print(f"caderno: {path}")
        print("arquivo atualizado. o chat ja le. LoRA so com: python -m agentepc learn --train")
        return

    if args.cmd == "learn":
        if args.train:
            print(json.dumps(train.run(force=args.force), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(train.plan(force=args.force), ensure_ascii=False, indent=2))
        return

    if args.cmd == "import-file":
        src = Path(args.file)
        dest = memory.write_knowledge(args.name or src.name, src.read_text(encoding="utf-8"))
        print(f"caderno: {dest}")
        print("consulta de arquivo ja ve isso. peso so depois de: python -m agentepc learn --train")


if __name__ == "__main__":
    main()
