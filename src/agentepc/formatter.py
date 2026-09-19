from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from agentepc import lotes, ollama
from agentepc.config import load, resolve

DEFAULT_SKILL = """# Skill: montar arquivo de treino e de consulta

Voce transforma dados brutos em linhas de fatos. O MESMO arquivo serve para dois usos:
- consulta: o chat le o arquivo inteiro no prompt, entao linhas curtas e sem repeticao cabem melhor;
- treino: cada linha vira pergunta e resposta para o LoRA, entao cada linha precisa ser um fato completo.

## Regras
1. Saida: SO linhas que comecam com "- ". Uma informacao por linha. Nada de titulo, explicacao ou comentario.
2. Frase completa e curta (ate 25 palavras), com o sujeito escrito em TODA linha (nome proprio), em terceira pessoa.
3. Nunca use "eu", "meu", "minha", "nosso". Troque pelo nome do sujeito.
4. Dado com rotulo vira "Rotulo: valor". Nome ou parentesco vira "A mae de Fulano se chama Maria" (sempre "O/A <parentesco> de <sujeito> se chama <nome>").
   Sobrenome NAO e parente: "eu me chamo Ana Lima" vira "O nome completo de Ana e Ana Lima".
5. Nao invente nada e nao complete o que falta. So o que esta no texto.
6. Ignore senha, token, chave de API, cartao, CPF e caminhos absolutos.
7. NUNCA repita as linhas dos exemplos desta skill. So vale o que esta no Trecho.
8. Extraia TODOS os fatos do trecho, ate os obvios. Titulo com uma frase de explicacao ja e fato.
9. "SEM FATOS" so quando o trecho inteiro for menu, indice, cookie ou rodape. Na duvida, extraia.

## Documentacao tecnica
- Comando, flag e caminho vao COPIADOS, sem traduzir: "- O comando `docker run -d` sobe o container em segundo plano".
- Uma linha por comando, flag ou conceito. Nao resuma a pagina inteira numa linha.
- O sujeito e a ferramenta ("Docker", "o comando docker ps"), nao "voce" nem "o usuario".
- GUARDE O CONTEXTO na propria linha: versao, modulo, sistema operacional, tipo de mudanca.
  Voce recebe "Secao:" com a trilha de titulos — use o que ela diz. Sem isso o fato fica solto.
  Ex.: Secao "O que ha de novo no Python 3.14 > Removidos > argparse" e o item "Remove os
  parametros type, choices e metavar de BooleanOptionalAction" viram:
  "- No Python 3.14 foram removidos os parametros type, choices e metavar de
  `argparse.BooleanOptionalAction`, descontinuados desde o Python 3.12".
- Fato longo pode ficar longo. Nao corte comando, nome de funcao nem a explicacao que da
  sentido; e melhor uma linha comprida e completa do que tres pela metade.
- Se a pagina fala de outra ferramenta junto, o fato pode ser sobre ela; escreva o nome dela.
- Linha que comeca com "[imagem]" descreve uma figura: vire fato do que a figura ensina.
- Ignore menu, indice, rodape, "edite esta pagina", cookies e link de navegacao.

## Exemplo
Sujeito: Ana
Texto: eu moro em Recife e meu cachorro e o Rex
Linhas:
- Ana mora em Recife
- O cachorro de Ana se chama Rex

Sujeito: Ana
Texto: me chamo Ana Lima, minha mae e a Rita, tenho um irmao, o Caio. Nasci em Natal.
Linhas:
- O nome completo de Ana e Ana Lima
- A mae de Ana se chama Rita
- O irmao de Ana se chama Caio
- Ana nasceu em Natal

Sujeito: Docker
Texto: - Guides\n- Manuals\n## Install Docker\nGet Docker Desktop or Docker Engine for your operating system.\n### docker ps\nLista os containers em execucao. Com -a mostra tambem os parados.
Linhas:
- O Docker Desktop e o Docker Engine sao as formas de instalar o Docker
- O comando `docker ps` lista os containers em execucao
- O comando `docker ps -a` lista tambem os containers parados
"""

_job: dict = {"state": "idle", "done": 0, "total": 0, "text": "", "error": "", "dropped": 0,
              "copied": 0, "facts": 0, "file": "", "big": False, "skipped": 0, "last_error": "",
              "cmds": [0, 0], "invented": 0, "lote": "", "assunto": "", "terminou": False,
              "apagou_bruto": False, "paginas": 0, "paginas_total": 0, "pagina": "",
              "retomado": 0}


def skill_path():
    path = resolve("data/skills/formato-de-treino.md")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_SKILL, encoding="utf-8")
    return path


def _norm(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9`]+", " ", text).strip()


# Tokens que nao mudam quando o modelo traduz: comandos, nomes proprios, versoes, siglas.
_DISTINTO = re.compile(
    r"`[^`]+`"                              # comando
    r"|\b[A-Z][\w.-]*[A-Z\d][\w.-]*\b"      # sigla ou CamelCase: WSL, Hyper-V, VMM
    r"|\b[\w.-]*\d[\w.-]*\b"                # versao ou numero: 2022, v2, 24.04
    r"|\b[A-Z][a-z]{2,}\b"                   # nome proprio: Docker, Kubernetes, Windows
)
_IGNORA = {"Uma", "Para", "Com", "Sem", "Por", "Nao", "Não", "Quando", "Depois", "Antes",
           "Este", "Esta", "Esse", "Essa", "Cada", "Todo", "Toda", "Onde", "Como", "Use"}


def grounded(line: str, fonte: str) -> bool:
    """O fato tem que estar NO TEXTO.

    A doc costuma estar em ingles e o fato sai em portugues, entao comparar palavra a
    palavra derrubaria traducao boa. O que se compara e o que sobrevive a traducao:
    comando entre crases, nome proprio, sigla e versao. Foi assim que apareceu o
    `docker build` numa pagina de instalacao que nao citava esse comando.
    """
    n_fonte = _norm(fonte)
    marcas = []
    for m in _DISTINTO.finditer(line):
        bruto = m.group(0).strip("`")
        if bruto in _IGNORA or len(bruto) < 2:
            continue
        marcas.append(bruto)
    for marca in marcas:
        n_marca = _norm(marca)
        if not n_marca or n_marca in n_fonte:
            continue
        # o modelo qualifica o nome com o modulo da secao ("argparse.BooleanOptionalAction");
        # isso vem do texto, so que em pedacos. Vale se cada pedaco estiver la.
        partes = [w for w in re.split(r"[^a-z0-9]+", n_marca) if len(w) > 2]
        if partes and all(w in n_fonte for w in partes):
            continue
        return False
    if marcas:
        return True
    # sem nada distintivo, sobra a comparacao por palavras
    palavras = {w for w in _norm(line).split() if len(w) >= 5}
    if not palavras:
        return False
    return sum(1 for w in palavras if w in n_fonte) / len(palavras) >= 0.5


def is_secret(line: str) -> bool:
    low = line.lower()
    never = (load().get("promote") or {}).get("never_bake") or []
    return any(tag in low for tag in never) or bool(
        re.search(r"\b(password|api[_ ]?key|secret|cpf|cartao|cartão)\b", low)
    )


def lint(text: str) -> list[str]:
    """Avisos para o usuario ajustar antes de treinar. Nao bloqueia."""
    out: list[str] = []
    seen: set[str] = set()
    facts = 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if is_secret(line):
            out.append(f"linha {n}: parece segredo/dado sensivel; sera ignorada no treino")
        if not line.startswith("- "):
            out.append(f"linha {n}: nao comeca com '- ' (o treino le fatos por linha)")
            continue
        facts += 1
        body = line[2:]
        if re.search(r"\b(eu|meu|minha|meus|minhas|nosso|nossa)\b", body.lower()):
            out.append(f"linha {n}: primeira pessoa; troque pelo nome (o modelo se confunde)")
        # linha longa NAO e defeito: comando e explicacao tecnica sao compridos mesmo,
        # e o modelo aprende a linha inteira. So o que nao chega a ser um fato incomoda.
        if len(body.split()) < 3:
            out.append(f"linha {n}: curta demais para virar pergunta ({body[:40]})")
        if low in seen:
            out.append(f"linha {n}: repetida")
        seen.add(low)
    if facts < int((load().get("learn") or {}).get("min_facts") or 5):
        out.append(f"so {facts} fato(s); o treino pede pelo menos {(load().get('learn') or {}).get('min_facts', 5)}")
    return out


_conserto: dict = {"state": "idle", "done": 0, "total": 0, "text": "", "mudou": 0, "tirou": 0}


def conserto_status() -> dict:
    return dict(_conserto)


def _curta(linha: str) -> bool:
    return len(linha[2:].split()) < 3


def _primeira_pessoa(linha: str) -> bool:
    return bool(re.search(r"\b(eu|meu|minha|meus|minhas|nosso|nossa)\b", linha.lower()))


def corrigir(texto: str, subject: str = "") -> dict:
    """Arruma o que o Conferir apontou: tira o que nao e fato e reescreve a 1a pessoa.

    Nao encurta linha comprida — comando e explicacao tecnica sao assim mesmo.
    """
    if _conserto["state"] == "running":
        return {"accepted": False, "reason": "ja estou consertando"}
    linhas = (texto or "").splitlines()
    if not linhas:
        return {"accepted": False, "reason": "nada para consertar"}
    _conserto.update({"state": "running", "done": 0, "total": len(linhas), "text": "",
                      "mudou": 0, "tirou": 0})
    skill = skill_path().read_text(encoding="utf-8")

    def _go() -> None:
        try:
            saida: list[str] = []
            vistos: set[str] = set()
            for linha in linhas:
                _conserto["done"] += 1
                bruta = linha.rstrip()
                if not bruta.startswith("- "):
                    saida.append(bruta)          # titulo e linha em branco ficam
                    continue
                chave = _norm(bruta)
                if chave in vistos or _curta(bruta) or is_secret(bruta):
                    _conserto["tirou"] += 1
                    continue
                vistos.add(chave)
                if _primeira_pessoa(bruta):
                    nova = _reescreve(bruta, subject, skill)
                    if nova and not _primeira_pessoa(nova):
                        _conserto["mudou"] += 1
                        saida.append(nova)
                        continue
                    _conserto["tirou"] += 1
                    continue
                saida.append(bruta)
            _conserto["text"] = "\n".join(saida).rstrip() + "\n"
            _conserto["state"] = "done"
        except Exception as exc:
            _conserto["state"] = "error"
            _conserto["text"] = str(exc)

    threading.Thread(target=_go, name="conserto", daemon=True).start()
    return {"accepted": True, "linhas": len(linhas)}


def _reescreve(linha: str, subject: str, skill: str) -> str:
    alvo = subject or "o sujeito do texto"
    try:
        data = ollama.request(
            "/api/chat",
            {
                "model": ollama.base_name(),
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0, "num_predict": 160},
                "messages": [
                    {"role": "system", "content": skill},
                    {"role": "user", "content": (
                        f"Reescreva esta linha em terceira pessoa, trocando eu/meu por {alvo}. "
                        f"Mantenha comando e detalhe tecnico exatamente como estao. "
                        f"Responda so a linha, comecando com '- '.\n{linha}"
                    )},
                ],
            },
            timeout=120,
        )
        for l in ((data.get("message") or {}).get("content") or "").splitlines():
            if l.strip().startswith("- "):
                return l.strip()
    except Exception:
        pass
    return ""


def _chunks(raw: str, size: int = 1200) -> list[tuple[str, str, str]]:
    """Devolve (trecho, trilha de titulos, pagina).

    Duas regras que o corte por tamanho sozinho quebrava:

    - **titulo fecha o trecho**. Antes, o fim da cota de caracteres podia cair no meio da
      explicacao de um modulo, e metade da informacao ia para outro lote, sem o contexto.
    - **pagina nao se mistura com pagina**. Cada `## titulo` do arquivo de extracao e uma
      pagina; o trecho nunca atravessa essa fronteira.
    """
    partes: list[tuple[str, str, str]] = []
    cur, trilha_ini, pagina, pagina_ini = "", [], "", ""
    pilha: list[tuple[int, str]] = []
    for linha in raw.splitlines():
        p = linha.strip()
        if not p:
            continue
        titulo = re.match(r"^(#{1,6})\s+(.+)$", p)
        if titulo:
            nivel = len(titulo.group(1))
            pilha = [(n, x) for n, x in pilha if n < nivel] + [(nivel, titulo.group(2).strip())]
            if nivel <= 2:
                pagina = titulo.group(2).strip()
        # qualquer titulo fecha o trecho anterior: assim nenhum assunto fica pela metade
        if cur and (titulo or len(cur) + len(p) > size):
            partes.append((cur, " > ".join(trilha_ini), pagina_ini))
            cur = ""
        if not cur:
            trilha_ini = [x for _, x in pilha]
            pagina_ini = pagina
        cur = (cur + "\n" + p).strip()
    if cur:
        partes.append((cur, " > ".join(trilha_ini), pagina_ini))

    def _tem_corpo(trecho: str) -> bool:
        corpo = [l for l in trecho.splitlines()
                 if l.strip() and not l.lstrip().startswith("#") and not l.startswith("fonte:")]
        return sum(len(l) for l in corpo) >= 80

    return [(c, tr, pg) for c, tr, pg in partes if _tem_corpo(c)]


def status() -> dict:
    return {**_job, "skill": skill_path().read_text(encoding="utf-8"), "skill_file": "data/skills/formato-de-treino.md"}


def stop() -> dict:
    if _job["state"] in ("running", "comandos"):
        _job["state"] = "parando"
    return status()


PREVIEW_MAX = 40000  # acima disso a pagina mostra amostra e salva direto do disco


def start(raw: str = "", subject: str = "", source_file: str = "", lote: str = "") -> dict:
    """Texto colado ou um lote de extracao. Um de cada vez, e o estado vive no servidor."""
    if _job["state"] in ("running", "parando", "comandos"):
        return {"accepted": False, "reason": "ja tem um lote sendo formatado"}
    from agentepc import crawler

    if crawler.status()["state"] == "running":
        return {"accepted": False, "reason": "tem uma busca rodando; espere ou pare a busca"}
    if lote and not source_file:
        meta = lotes.ler_meta(lote)
        alvo = lotes.pasta() / f"{lote}.md"
        if not alvo.exists():
            return {"accepted": False, "reason": "esse lote nao tem texto extraido"}
        source_file = str(alvo)
        subject = subject or meta.get("assunto") or meta.get("nome") or ""
    if source_file:
        path = Path(source_file)
        if not path.is_absolute():
            path = resolve(source_file)
        if not path.exists():
            return {"accepted": False, "reason": "arquivo de extracao nao encontrado"}
        raw = path.read_text(encoding="utf-8", errors="replace")
    raw = (raw or "").strip()
    if not raw:
        return {"accepted": False, "reason": "cole os dados ou extraia um site antes"}
    subject = (subject or "").strip() or "(descubra no texto quem e o sujeito principal e escreva o nome dele em toda linha)"
    chunks = _chunks(raw)
    base = lote or (Path(source_file).stem if source_file else f"colado-{int(time.time())}")
    out_path = lotes.pasta() / f"{base}-fatos.md"
    progresso = lotes.pasta() / f"{base}.progresso.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cabecalho = f"# Fatos{(' sobre ' + subject) if not subject.startswith('(') else ''}\n\n"
    # parou no meio? continua de onde estava, em vez de refazer tudo
    feito_antes, fatos_antes = 0, 0
    if progresso.exists() and out_path.exists():
        try:
            marca = json.loads(progresso.read_text(encoding="utf-8"))
            if marca.get("total") == len(chunks):
                feito_antes, fatos_antes = int(marca.get("done", 0)), int(marca.get("facts", 0))
        except (OSError, ValueError):
            feito_antes = 0
    if not feito_antes:
        out_path.write_text(cabecalho, encoding="utf-8")
    paginas_total = len({pg for _, _, pg in chunks if pg})
    _job.update({"state": "running", "done": feito_antes, "total": len(chunks), "text": "",
                 "error": "", "dropped": 0, "copied": 0, "facts": fatos_antes,
                 "file": str(out_path), "big": False, "skipped": 0, "last_error": "",
                 "cmds": [0, 0], "invented": 0,
                 "lote": lote or (Path(source_file).stem if source_file else ""),
                 "assunto": subject, "terminou": False, "apagou_bruto": False,
                 "paginas": 0, "paginas_total": paginas_total, "pagina": "",
                 "retomado": feito_antes})
    skill = skill_path().read_text(encoding="utf-8")
    # o modelo pequeno as vezes copia a resposta do exemplo; essas linhas nao vieram do texto
    exemplos = {_norm(ln) for ln in skill.splitlines() if ln.strip().startswith("- ")}

    def _ask(chunk: str, insistir: bool, contexto: str = "") -> str:
        cabeca = f"Secao: {contexto}\n" if contexto else ""
        pedido = (
            f"Sujeito: {subject}\n{cabeca}"
            "REGRA: uma linha para CADA item do texto. Comece pelo contexto da secao "
            "(versao, modulo, sistema) e depois o fato completo, com os nomes tecnicos "
            "exatamente como aparecem.\n"
            f"Texto: {chunk}\nLinhas:"
        )
        if insistir:
            pedido = (
                "Este trecho TEM fatos. Extraia pelo menos um por titulo ou comando. Nao responda SEM FATOS.\n"
                + pedido
            )
        data = ollama.request(
            "/api/chat",
            {
                "model": ollama.base_name(),
                "stream": False,
                "keep_alive": "10m",
                # sem teto o modelo pequeno as vezes entra em loop e estoura o tempo do lote
                "options": {"temperature": 0, "num_predict": 400},
                "messages": [
                    {"role": "system", "content": skill},
                    {"role": "user", "content": pedido},
                ],
            },
            timeout=300,
        )
        return (data.get("message") or {}).get("content") or ""

    def _ask_safe(chunk: str, insistir: bool, contexto: str = "") -> str:
        """Um lote problematico e pulado; num site inteiro isso nao pode perder o resto."""
        for tentativa in (1, 2):
            try:
                return _ask(chunk, insistir, contexto)
            except Exception as exc:
                if tentativa == 2:
                    _job["skipped"] += 1
                    _job["last_error"] = str(exc)[:200]
                    return ""
        return ""

    def _cobrir(fh, texto: str, vistos: set[str]) -> None:
        """Segunda passada so nos comandos que nao viraram fato.

        Doc existe por causa dos comandos; se um escapou, a IA fica sem metade da ferramenta.
        """
        from agentepc.crawler import commands

        todos = commands(texto)
        if not todos:
            return
        ja = " ".join(vistos)
        faltando = [c for c in todos if c.lower() not in ja][:60]
        _job["cmds"] = [len(todos), len(todos) - len(faltando)]
        for cmd in faltando:
            if _job["state"] == "parando":
                break
            pos = texto.lower().find(cmd.lower())
            trecho = texto[max(0, pos - 700): pos + 700] if pos >= 0 else ""
            saida = _ask_safe(
                f"{trecho}\n\nEscreva UMA linha, comecando com \"- \", dizendo o que o comando "
                f"`{cmd}` faz segundo o trecho. Se o trecho nao diz, responda SEM FATOS.",
                insistir=False,
            )
            for line in saida.splitlines():
                line = line.strip()
                if (line.startswith("- ") and line.lower() not in vistos
                        and not is_secret(line) and grounded(line, trecho)):
                    vistos.add(line.lower())
                    fh.write(line + "\n")
                    _job["facts"] += 1
                    _job["cmds"][1] += 1
                    break
        fh.flush()

    def _go() -> None:
        try:
            vistos: set[str] = set()
            paginas_vistas: set[str] = set()
            with out_path.open("a", encoding="utf-8") as fh:
                for indice, (chunk, contexto, pagina) in enumerate(chunks):
                    if indice < feito_antes:
                        continue          # ja formatado numa passada anterior
                    if _job["state"] == "parando":
                        break
                    if pagina and pagina not in paginas_vistas:
                        paginas_vistas.add(pagina)
                        _job["paginas"] = len(paginas_vistas)
                        _job["pagina"] = pagina
                    saida = _ask_safe(chunk, insistir=False, contexto=contexto)
                    # trecho grande sem nenhum fato quase sempre e o modelo sendo conservador
                    if "- " not in saida and len(chunk) > 400:
                        saida = _ask_safe(chunk, insistir=True, contexto=contexto)
                    for line in saida.splitlines():
                        line = line.strip()
                        if not line.startswith("- ") or line.lower() in vistos:
                            continue
                        if _norm(line) in exemplos:
                            _job["copied"] += 1  # veio do exemplo da skill, nao do texto
                            continue
                        if is_secret(line):
                            _job["dropped"] += 1  # segredo nunca vai para o arquivo de treino
                            continue
                        if not grounded(line, chunk + "\n" + contexto):
                            _job["invented"] += 1  # nao esta no texto: seria treinar invencao
                            continue
                        vistos.add(line.lower())
                        fh.write(line + "\n")
                        _job["facts"] += 1
                    fh.flush()
                    _job["done"] += 1
                    progresso.write_text(json.dumps(
                        {"done": _job["done"], "total": len(chunks), "facts": _job["facts"]}),
                        encoding="utf-8")
                _job["state"] = "comandos" if _job["state"] == "running" else _job["state"]
                _cobrir(fh, raw, vistos)
            tamanho = out_path.stat().st_size
            _job["big"] = tamanho > PREVIEW_MAX
            texto = out_path.read_text(encoding="utf-8")
            _job["text"] = texto if not _job["big"] else texto[:PREVIEW_MAX]
            # o texto bruto ja cumpriu o papel; fica so o arquivo de fatos
            if _job["state"] != "parando" and _job["lote"] and _job["facts"]:
                lotes.apagar(_job["lote"], so_bruto=True)
                progresso.unlink(missing_ok=True)
                _job["apagou_bruto"] = True
            _job["state"] = "done"
            _job["terminou"] = True
        except Exception as exc:
            _job["state"] = "error"
            _job["error"] = str(exc)

    threading.Thread(target=_go, name="formatter", daemon=True).start()
    return {"accepted": True, "chunks": len(chunks), "file": str(out_path)}
