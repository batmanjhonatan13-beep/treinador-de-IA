"""Conhecimento geral: o que o modelo ja sabia e o treino nao pode destruir.

Um LoRA deveria ser conhecimento A MAIS. So que treinar 8 fatos ate a prova fechar tres
vezes seguidas significa dezenas de passadas sobre um punhado de exemplos parecidos — e o
que o modelo aprende ali nao e "mais um assunto", e "responda com algo deste arquivo, seja
qual for a pergunta". A literatura chama isso de esquecimento catastrofico; na pratica voce
pergunta a capital da Franca e ele responde o nome do seu servidor.

Este modulo resolve por dois lados:

  1. REPLAY — junto dos exemplos do seu arquivo entram exemplos do conhecimento geral, com
     as respostas que o PROPRIO modelo base da. Assim o treino tem que preservar o velho
     enquanto aprende o novo, em vez de passar por cima.
  2. PROVA — no fim do treino as mesmas perguntas gerais sao refeitas com o adaptador
     ligado. Se o numero cair, aparece na tela. Sem medir, o estrago passa despercebido:
     a prova do arquivo continua 8/8 e parece que esta tudo bem.

As respostas certas nao sao escritas por mim: sao colhidas do modelo base. O alvo nao e
"acertar a capital da Franca", e "continuar respondendo o que voce respondia antes".
"""

from __future__ import annotations

import json
from pathlib import Path

from agentepc.config import load, resolve
from agentepc.prompting import SYSTEM

# Perguntas de tres tipos, porque o treino estraga os tres:
#   - fato comum (o modelo troca a resposta pelo seu fato)
#   - saber tecnico (o que voce usa no dia a dia)
#   - o "nao sei" (o modelo passa a chutar o seu fato em vez de admitir que nao sabe)
# Perguntas de tres tipos, porque o treino estraga os tres: fato comum, saber tecnico, e o
# "nao sei" — este ultimo e o que mais se perde. Depois de decorar um arquivo, o modelo
# para de admitir que nao sabe e passa a chutar um fato seu em qualquer pergunta.
PERGUNTAS = [
    {"q": "Qual e a capital da Franca?"},
    {"q": "Qual e a capital do Brasil?"},
    {"q": "Quanto e 7 vezes 8?"},
    {"q": "Quanto e 25 mais 17?"},
    {"q": "Quem escreveu Dom Casmurro?"},
    {"q": "Em que ano o homem pisou na Lua pela primeira vez?"},
    {"q": "Qual e o maior planeta do sistema solar?"},
    {"q": "Qual e a formula quimica da agua?"},
    {"q": "Quantos dias tem um ano bissexto?"},
    {"q": "Como se diz obrigado em ingles?"},
    {"q": "Qual e o idioma falado na Argentina?"},
    {"q": "Qual e o oceano que banha o litoral brasileiro?"},
    {"q": "Quem pintou a Mona Lisa?"},
    {"q": "Qual e o animal terrestre mais rapido?"},
    {"q": "Quantos lados tem um hexagono?"},
    {"q": "O que significa a sigla HTTP?"},
    {"q": "O que o comando ls faz no Linux?"},
    {"q": "Qual comando lista os containers em execucao no Docker?"},
    {"q": "Para que serve o comando git commit?"},
    {"q": "O que e uma chave primaria num banco de dados?"},
    {"q": "Qual porta o HTTPS usa por padrao?"},
    {"q": "O que significa a sigla DNS?"},
    {"q": "Para que serve o arquivo requirements.txt em Python?"},
    {"q": "O que o comando chmod faz?"},
    {"q": "Qual e a diferenca entre RAM e disco?"},
    {"q": "O que e um container?"},
    {"q": "Qual e o numero do meu documento de identidade?", "recusa": True},
    {"q": "Qual e a senha do meu e-mail?", "recusa": True},
    {"q": "Quantos parafusos tem a minha cadeira?", "recusa": True},
    {"q": "Que horas eu almocei ontem?", "recusa": True},
]

# quantas perguntas gerais entram no treino junto com as suas
PADRAO_REPLAY = 30


def arquivo() -> Path:
    return resolve("data/geral.jsonl")


def _modelo_atual() -> str:
    cfg = load().get("model") or {}
    return str(cfg.get("base_hf") or cfg.get("name") or "?")


def construir(force: bool = False, diz=None) -> list[dict]:
    """Colhe do modelo base as respostas dele e guarda em disco.

    Uma vez so por modelo: trocar de modelo invalida o arquivo, porque o que se quer
    preservar e o comportamento DAQUELE modelo, nao de um outro qualquer.
    """
    from agentepc import ollama

    alvo = arquivo()
    if alvo.exists() and not force:
        dados = [json.loads(l) for l in alvo.read_text(encoding="utf-8").splitlines() if l.strip()]
        if dados and dados[0].get("modelo") == _modelo_atual():
            return dados
    fala = diz or (lambda _m: None)
    fala(f"colhendo o conhecimento geral de {_modelo_atual()} (uma vez so)")
    from agentepc import exam

    linhas = []
    descartadas = 0
    for item in PERGUNTAS:
        pergunta = item["q"]
        try:
            resposta = ollama.request("/api/chat", {
                "model": ollama.model_name(),
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": pergunta}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 48},
            })
            texto = ((resposta.get("message") or {}).get("content") or "").strip()
        except Exception as exc:
            fala(f"nao consegui perguntar ao modelo base: {exc}")
            break
        texto = " ".join(texto.split())[:200]
        if not texto:
            continue
        recusou = exam._refuse(texto)
        if item.get("recusa"):
            # aqui recusar E a resposta certa: e esse habito que o treino costuma matar
            if not recusou:
                descartadas += 1
                continue
        elif recusou:
            # o base nao soube responder isto; nao serve nem de exemplo nem de regua
            descartadas += 1
            continue
        linhas.append({"modelo": _modelo_atual(), "user": pergunta, "assistant": texto,
                       "recusa": bool(item.get("recusa"))})
    if descartadas:
        fala(f"{descartadas} pergunta(s) fora: o modelo base nao respondeu de forma util")
    if linhas:
        alvo.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in linhas),
                        encoding="utf-8")
        fala(f"{len(linhas)} respostas do modelo base guardadas em {alvo.name}")
    return linhas


def pares(limite: int | None = None, diz=None) -> list[dict]:
    """Exemplos de replay para entrar no treino junto com os seus fatos."""
    if limite is None:
        # cuidado com o zero: "x or padrao" trocaria replay:0 pelo padrao, e quem desligou
        # o replay continuaria treinando com ele
        bruto = (load().get("learn") or {}).get("replay")
        limite = PADRAO_REPLAY if bruto is None else int(bruto)
    if limite <= 0:
        return []
    dados = construir(diz=diz)
    return [{"user": d["user"], "assistant": d["assistant"]} for d in dados[:limite]]


def prova(respostas: list[str]) -> dict:
    """Compara as respostas de agora com as do modelo base, pergunta a pergunta.

    O gabarito e a resposta que o base deu. Uso as mesmas palavras-chave da prova do
    arquivo: se as palavras que importavam sumiram, o conhecimento foi embora.
    """
    from agentepc import exam

    base = construir()
    if not base or len(respostas) < len(base):
        return {"total": 0, "mantidas": 0, "perdidas": []}
    perdidas = []
    mantidas = 0
    for esperado, obtida in zip(base, respostas):
        if esperado.get("recusa"):
            # o que se cobra aqui e o habito de admitir que nao sabe, nao a frase exata
            certo = exam._refuse(obtida or "")
        else:
            certo = exam.passed(obtida, exam.keys_for(esperado["assistant"]))
        if certo:
            mantidas += 1
        else:
            perdidas.append({"q": esperado["user"], "antes": esperado["assistant"][:80],
                             "agora": (obtida or "")[:80]})
    return {"total": len(base), "mantidas": mantidas, "perdidas": perdidas[:6]}


def perguntas() -> list[str]:
    return [d["user"] for d in construir()]


def resumo(medida: dict) -> str:
    """Uma linha para o log e para a tela."""
    if not medida.get("total"):
        return "conhecimento geral: nao medido"
    n, t = medida["mantidas"], medida["total"]
    linha = f"conhecimento geral: {n}/{t} mantidos"
    if n < t:
        exemplo = medida["perdidas"][0]
        linha += (f" — perdeu, por exemplo: \"{exemplo['q']}\" respondia "
                  f"\"{exemplo['antes']}\" e agora responde \"{exemplo['agora']}\"")
    return linha
