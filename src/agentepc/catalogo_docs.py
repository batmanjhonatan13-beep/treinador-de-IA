"""Onde baixar documentacao pronta, por ferramenta.

Andar pelo site pagina a pagina e o pior caminho: demora, incomoda o servidor e muitos
sites cortam o acesso no meio (a tela de "Um momento..."). Quase tudo que importa tem um
pacote pronto — so nao esta no mesmo lugar para todo mundo. Este arquivo e essa lista.

Cada item tem:
  * `pacote` — endereco de um .zip/.tar.gz com a doc inteira, quando existe. Um download.
  * `doc` — o endereco oficial, para quando nao ha pacote: dai vale a busca por paginas.

Os enderecos foram conferidos um a um (HTTP 200 e o tipo do arquivo). Versao muda: se um
pacote der erro, o `doc` continua valendo e o site costuma ter o link novo.

Sobre os conjuntos do Dash (kapeli.com): sao copias da documentacao oficial mantidas pela
Kapeli e pela comunidade, feitas para leitura offline. A licenca de cada documentacao
continua sendo a do projeto de origem — se voce for publicar algo treinado nelas, e essa
licenca que vale, nao a do pacote.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "agente-pc/1.0 (catalogo de documentacao)"}


def _dash(nome: str) -> str:
    return f"https://kapeli.com/feeds/{nome}.tgz"


def _dash_com(nome: str) -> str:
    return f"https://kapeli.com/feeds/zzz/user_contributed/build/{nome}/{nome}.tgz"


CATEGORIAS = {
    "linguagem": "Linguagens",
    "dev": "Ferramentas de desenvolvimento",
    "dados": "Bancos e mensageria",
    "devops": "DevOps e CI/CD",
    "sre": "SRE e observabilidade",
    "design": "Design, arte e 3D",
}

CATALOGO: list[dict] = [
    # ---------------------------------------------------------------- linguagens
    {"id": "python-ptbr", "nome": "Python", "versao": "3.14", "categoria": "linguagem",
     "pacote": "https://docs.python.org/pt-br/3/archives/python-3.14-docs-html.zip",
     "doc": "https://docs.python.org/pt-br/3/", "nota": "em português"},
    {"id": "python-en", "nome": "Python (inglês)", "versao": "3.14", "categoria": "linguagem",
     "pacote": "https://docs.python.org/3/archives/python-3.14-docs-html.zip",
     "doc": "https://docs.python.org/3/", "nota": "571 páginas no teste daqui"},
    {"id": "go", "nome": "Go", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Go"), "doc": "https://go.dev/doc/", "nota": ""},
    {"id": "rust", "nome": "Rust", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Rust"), "doc": "https://doc.rust-lang.org/book/", "nota": ""},
    {"id": "bash", "nome": "Bash", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Bash"), "doc": "https://www.gnu.org/software/bash/manual/", "nota": ""},
    {"id": "javascript", "nome": "JavaScript", "versao": "", "categoria": "linguagem",
     "pacote": _dash("JavaScript"), "doc": "https://developer.mozilla.org/pt-BR/docs/Web/JavaScript", "nota": ""},
    {"id": "typescript", "nome": "TypeScript", "versao": "", "categoria": "linguagem",
     "pacote": _dash("TypeScript"), "doc": "https://www.typescriptlang.org/docs/", "nota": ""},
    {"id": "java", "nome": "Java", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Java"), "doc": "https://docs.oracle.com/en/java/", "nota": ""},
    {"id": "php", "nome": "PHP", "versao": "", "categoria": "linguagem",
     "pacote": _dash("PHP"), "doc": "https://www.php.net/manual/pt_BR/", "nota": ""},
    {"id": "ruby", "nome": "Ruby", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Ruby"), "doc": "https://www.ruby-lang.org/pt/documentation/", "nota": ""},
    {"id": "c", "nome": "C", "versao": "", "categoria": "linguagem",
     "pacote": _dash("C"), "doc": "https://en.cppreference.com/w/c", "nota": ""},
    {"id": "cpp", "nome": "C++", "versao": "", "categoria": "linguagem",
     "pacote": _dash("C++"), "doc": "https://en.cppreference.com/w/cpp", "nota": ""},
    {"id": "swift", "nome": "Swift", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Swift"), "doc": "https://www.swift.org/documentation/", "nota": ""},
    {"id": "lua", "nome": "Lua", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Lua"), "doc": "https://www.lua.org/manual/5.4/", "nota": ""},
    {"id": "perl", "nome": "Perl", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Perl"), "doc": "https://perldoc.perl.org/", "nota": ""},
    {"id": "haskell", "nome": "Haskell", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Haskell"), "doc": "https://www.haskell.org/documentation/", "nota": ""},
    {"id": "r", "nome": "R", "versao": "", "categoria": "linguagem",
     "pacote": _dash("R"), "doc": "https://cran.r-project.org/manuals.html", "nota": ""},
    {"id": "elixir", "nome": "Elixir", "versao": "", "categoria": "linguagem",
     "pacote": _dash("Elixir"), "doc": "https://elixir-lang.org/docs.html", "nota": ""},
    {"id": "nodejs", "nome": "Node.js", "versao": "", "categoria": "linguagem",
     "pacote": "", "doc": "https://nodejs.org/docs/latest/api/",
     "nota": "sem pacote pronto — use a busca por páginas"},
    {"id": "kotlin", "nome": "Kotlin", "versao": "", "categoria": "linguagem",
     "pacote": "", "doc": "https://kotlinlang.org/docs/home.html",
     "nota": "sem pacote pronto"},

    # ---------------------------------------------------- ferramentas de desenvolvimento
    {"id": "git", "nome": "Git", "versao": "2.47", "categoria": "dev",
     "pacote": "https://mirrors.edge.kernel.org/pub/software/scm/git/git-htmldocs-2.47.0.tar.gz",
     "doc": "https://git-scm.com/docs", "nota": "tarball oficial do kernel.org"},
    {"id": "vim", "nome": "Vim", "versao": "", "categoria": "dev",
     "pacote": _dash("Vim"), "doc": "https://vimhelp.org/", "nota": ""},
    {"id": "django", "nome": "Django", "versao": "5.2", "categoria": "dev",
     "pacote": "https://media.djangoproject.com/docs/django-docs-5.2-en.zip",
     "doc": "https://docs.djangoproject.com/pt-br/", "nota": "zip oficial do projeto"},
    {"id": "flask", "nome": "Flask", "versao": "", "categoria": "dev",
     "pacote": _dash("Flask"), "doc": "https://flask.palletsprojects.com/", "nota": ""},
    {"id": "react", "nome": "React", "versao": "", "categoria": "dev",
     "pacote": _dash("React"), "doc": "https://react.dev/reference/react", "nota": ""},
    {"id": "angular", "nome": "Angular", "versao": "", "categoria": "dev",
     "pacote": _dash("Angular"), "doc": "https://angular.dev/overview", "nota": ""},
    {"id": "github-actions", "nome": "GitHub Actions", "versao": "", "categoria": "dev",
     "pacote": "", "doc": "https://docs.github.com/en/actions",
     "nota": "sem pacote pronto"},

    # ---------------------------------------------------------------- dados
    {"id": "postgresql", "nome": "PostgreSQL", "versao": "", "categoria": "dados",
     "pacote": _dash("PostgreSQL"), "doc": "https://www.postgresql.org/docs/current/", "nota": ""},
    {"id": "mysql", "nome": "MySQL", "versao": "", "categoria": "dados",
     "pacote": _dash("MySQL"), "doc": "https://dev.mysql.com/doc/", "nota": ""},
    {"id": "sqlite", "nome": "SQLite", "versao": "", "categoria": "dados",
     "pacote": _dash("SQLite"), "doc": "https://www.sqlite.org/docs.html", "nota": ""},
    {"id": "mongodb", "nome": "MongoDB", "versao": "", "categoria": "dados",
     "pacote": _dash("MongoDB"), "doc": "https://www.mongodb.com/docs/", "nota": ""},
    {"id": "redis", "nome": "Redis", "versao": "", "categoria": "dados",
     "pacote": _dash("Redis"), "doc": "https://redis.io/docs/latest/", "nota": ""},
    {"id": "kafka", "nome": "Kafka", "versao": "0.11", "categoria": "dados",
     "pacote": _dash_com("kafka"), "doc": "https://kafka.apache.org/documentation/",
     "nota": "pacote antigo; a doc oficial está atual"},
    {"id": "elasticsearch", "nome": "Elasticsearch", "versao": "", "categoria": "dados",
     "pacote": "", "doc": "https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html",
     "nota": "sem pacote pronto"},

    # ---------------------------------------------------------------- devops
    {"id": "docker", "nome": "Docker", "versao": "", "categoria": "devops",
     "pacote": _dash("Docker"), "doc": "https://docs.docker.com/",
     "nota": "196 MB · 1 551 páginas no teste daqui — o site não publica pacote"},
    {"id": "kubernetes", "nome": "Kubernetes", "versao": "1.34", "categoria": "devops",
     "pacote": _dash_com("Kubernetes"), "doc": "https://kubernetes.io/docs/home/", "nota": ""},
    {"id": "terraform", "nome": "Terraform", "versao": "1.15", "categoria": "devops",
     "pacote": _dash_com("Terraform"), "doc": "https://developer.hashicorp.com/terraform/docs", "nota": ""},
    {"id": "ansible", "nome": "Ansible", "versao": "", "categoria": "devops",
     "pacote": _dash("Ansible"), "doc": "https://docs.ansible.com/", "nota": ""},
    {"id": "vault", "nome": "Vault", "versao": "1.8", "categoria": "devops",
     "pacote": _dash_com("Vault"), "doc": "https://developer.hashicorp.com/vault/docs", "nota": ""},
    {"id": "consul", "nome": "Consul", "versao": "1.10", "categoria": "devops",
     "pacote": _dash_com("Consul"), "doc": "https://developer.hashicorp.com/consul/docs", "nota": ""},
    {"id": "nomad", "nome": "Nomad", "versao": "1.1", "categoria": "devops",
     "pacote": _dash_com("Nomad"), "doc": "https://developer.hashicorp.com/nomad/docs", "nota": ""},
    {"id": "packer", "nome": "Packer", "versao": "1.7", "categoria": "devops",
     "pacote": _dash_com("Packer"), "doc": "https://developer.hashicorp.com/packer/docs", "nota": ""},
    {"id": "nginx", "nome": "Nginx", "versao": "", "categoria": "devops",
     "pacote": _dash("Nginx"), "doc": "https://nginx.org/en/docs/", "nota": ""},
    {"id": "envoy", "nome": "Envoy", "versao": "", "categoria": "devops",
     "pacote": _dash_com("Envoy"), "doc": "https://www.envoyproxy.io/docs", "nota": ""},
    {"id": "airflow", "nome": "Airflow", "versao": "", "categoria": "devops",
     "pacote": _dash_com("Airflow"), "doc": "https://airflow.apache.org/docs/", "nota": ""},
    {"id": "jenkins-jjb", "nome": "Jenkins Job Builder", "versao": "2.0", "categoria": "devops",
     "pacote": _dash_com("Jenkins_Job_Builder"), "doc": "https://www.jenkins.io/doc/",
     "nota": "o pacote cobre o Job Builder; o Jenkins inteiro só pela doc"},
    {"id": "aws-cli", "nome": "AWS CLI", "versao": "", "categoria": "devops",
     "pacote": _dash_com("AWS_CLI"), "doc": "https://docs.aws.amazon.com/cli/", "nota": ""},
    {"id": "aws-cfn", "nome": "AWS CloudFormation", "versao": "", "categoria": "devops",
     "pacote": _dash_com("AWS_CloudFormation_Template_Reference"),
     "doc": "https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/", "nota": ""},
    {"id": "helm", "nome": "Helm", "versao": "", "categoria": "devops",
     "pacote": "", "doc": "https://helm.sh/docs/", "nota": "sem pacote pronto"},
    {"id": "gitlab", "nome": "GitLab", "versao": "", "categoria": "devops",
     "pacote": "", "doc": "https://docs.gitlab.com/", "nota": "sem pacote pronto"},
    {"id": "argocd", "nome": "Argo CD", "versao": "", "categoria": "devops",
     "pacote": "", "doc": "https://argo-cd.readthedocs.io/en/stable/", "nota": "sem pacote pronto"},

    # ---------------------------------------------------------------- SRE
    {"id": "prometheus", "nome": "Prometheus", "versao": "", "categoria": "sre",
     "pacote": "", "doc": "https://prometheus.io/docs/introduction/overview/",
     "nota": "sem pacote pronto"},
    {"id": "grafana", "nome": "Grafana", "versao": "", "categoria": "sre",
     "pacote": "", "doc": "https://grafana.com/docs/grafana/latest/", "nota": "sem pacote pronto"},
    {"id": "opentelemetry", "nome": "OpenTelemetry", "versao": "", "categoria": "sre",
     "pacote": "", "doc": "https://opentelemetry.io/docs/", "nota": "sem pacote pronto"},

    # ---------------------------------------------------------------- design e 3D
    {"id": "blender", "nome": "Blender (manual)", "versao": "", "categoria": "design",
     "pacote": "https://docs.blender.org/manual/en/latest/blender_manual_html.zip",
     "doc": "https://docs.blender.org/manual/pt/latest/", "nota": "zip oficial do projeto"},
    {"id": "blender-dash", "nome": "Blender (API Python)", "versao": "4.2", "categoria": "design",
     "pacote": _dash_com("Blender"), "doc": "https://docs.blender.org/api/current/", "nota": ""},
    {"id": "godot", "nome": "Godot", "versao": "4.6", "categoria": "design",
     "pacote": _dash_com("Godot"), "doc": "https://docs.godotengine.org/pt-br/4.x/",
     "nota": "o site bloqueia varredura; use o pacote"},
    {"id": "threejs", "nome": "Three.js", "versao": "r179", "categoria": "design",
     "pacote": _dash_com("Three.js"), "doc": "https://threejs.org/docs/", "nota": ""},
    {"id": "inkscape", "nome": "Inkscape", "versao": "", "categoria": "design",
     "pacote": "https://inkscape-manuals.readthedocs.io/_/downloads/en/latest/htmlzip/",
     "doc": "https://inkscape.org/learn/", "nota": "manual da comunidade, no Read the Docs"},
    {"id": "gimp", "nome": "GIMP", "versao": "2.10", "categoria": "design",
     "pacote": "", "doc": "https://docs.gimp.org/2.10/pt_BR/",
     "nota": "o projeto publica pacotes por idioma em download.gimp.org/gimp/help/"},
    {"id": "krita", "nome": "Krita", "versao": "", "categoria": "design",
     "pacote": "", "doc": "https://docs.krita.org/en/", "nota": "sem pacote pronto"},
    {"id": "opencv", "nome": "OpenCV", "versao": "4.x", "categoria": "design",
     "pacote": "", "doc": "https://docs.opencv.org/4.x/", "nota": "sem pacote pronto"},
    {"id": "unreal", "nome": "Unreal Engine 4", "versao": "", "categoria": "design",
     "pacote": _dash_com("UnrealEngine4"), "doc": "https://dev.epicgames.com/documentation/",
     "nota": "pacote é da versão 4"},
]


# ------------------------------------------------------------------ atualizacao
# Versao envelhece: quando escrevi esta lista o Git estava na 2.47 e uma hora depois ja
# era 2.55. Em vez de voce conferir um por um, cada fonte sabe descobrir a propria versao.
DASH_XML = "https://kapeli.com/feeds/{}.xml"
DASH_COM_INDEX = "https://kapeli.com/feeds/zzz/user_contributed/build/index.json"

_atualizacao: dict = {"state": "idle", "linhas": [], "quando": "", "mudou": 0}


def _busca(url: str, timeout: int = 25) -> str:
    pedido = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(pedido, timeout=timeout) as r:
        return r.read(8_000_000).decode("utf-8", "replace")


def _cache() -> Path:
    from agentepc.config import resolve

    return resolve("data/catalogo-versoes.json")


def versoes() -> dict:
    arq = _cache()
    if arq.exists():
        try:
            return json.loads(arq.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def _fonte(item: dict) -> str:
    """Descobre sozinho de onde veio o pacote, pelo proprio endereco."""
    pacote = item.get("pacote") or ""
    if "user_contributed" in pacote:
        return "dash_com"
    if "kapeli.com/feeds/" in pacote:
        return "dash"
    if "docs.python.org" in pacote:
        return "python"
    if "djangoproject.com" in pacote:
        return "django"
    if "scm/git/" in pacote:
        return "git"
    return ""


def _limpa_versao(bruta: str) -> str:
    """Nem todo feed traz versao de verdade.

    Alguns conjuntos do Dash publicam numero de build no lugar ("/9", "/157"), e outros
    grudam um hash atras da versao ("1.1.2/001-6cba3c5c"). Mostrar isso ao usuario e pior
    do que nao mostrar nada: parece versao e nao e.
    """
    versao = (bruta or "").strip().lstrip("/").split("/")[0].strip()
    if not versao or re.fullmatch(r"\d{1,3}", versao):
        return ""
    return versao[:24]


def _versao_dash(item: dict) -> tuple[str, str]:
    nome = (item["pacote"].rsplit("/", 1)[-1]).replace(".tgz", "")
    xml = _busca(DASH_XML.format(nome))
    achado = re.search(r"<version>([^<]+)</version>", xml)
    return _limpa_versao(achado.group(1) if achado else ""), item["pacote"]


def _versao_dash_com(item: dict, indice: dict) -> tuple[str, str]:
    nome = (item["pacote"].rsplit("/", 1)[-1]).replace(".tgz", "")
    dados = (indice.get("docsets") or indice).get(nome) or {}
    return _limpa_versao(str(dados.get("version") or "")), item["pacote"]


def _versao_python(item: dict) -> tuple[str, str]:
    pagina = _busca("https://docs.python.org/pt-br/3/download.html")
    achado = re.search(r"archives/python-([0-9.]+)-docs-html\.zip", pagina)
    if not achado:
        return "", item["pacote"]
    versao = achado.group(1)
    base = "pt-br/3" if "pt-br" in item["pacote"] else "3"
    return versao, f"https://docs.python.org/{base}/archives/python-{versao}-docs-html.zip"


def _versao_django(item: dict) -> tuple[str, str]:
    dados = json.loads(_busca("https://pypi.org/pypi/Django/json"))
    cheia = dados["info"]["version"]
    curta = ".".join(cheia.split(".")[:2])
    return curta, f"https://media.djangoproject.com/docs/django-docs-{curta}-en.zip"


def _versao_git(item: dict) -> tuple[str, str]:
    pagina = _busca("https://mirrors.edge.kernel.org/pub/software/scm/git/")
    achados = re.findall(r"git-htmldocs-([0-9][0-9.]*)\.tar\.gz", pagina)
    if not achados:
        return "", item["pacote"]
    versao = sorted(achados, key=lambda v: [int(x) for x in v.strip(".").split(".")])[-1]
    return versao, ("https://mirrors.edge.kernel.org/pub/software/scm/git/"
                    f"git-htmldocs-{versao}.tar.gz")


def status_atualizacao() -> dict:
    return {**_atualizacao, "quando": versoes().get("_quando", "")}


def atualizar() -> dict:
    """Confere a versao de cada pacote e guarda o que mudou.

    Roda em segundo plano porque sao dezenas de pedidos. O que falhar fica com o valor
    antigo — um pacote fora do ar nao pode derrubar a lista inteira.
    """
    if _atualizacao["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma atualizacao rodando"}
    _atualizacao.update({"state": "running", "linhas": [], "mudou": 0})

    def _diz(msg: str) -> None:
        _atualizacao["linhas"].append(msg)
        del _atualizacao["linhas"][:-40]

    def _go() -> None:
        try:
            achado = versoes()
            achado.pop("_quando", None)
            indice = {}
            if any(_fonte(i) == "dash_com" for i in CATALOGO):
                try:
                    indice = json.loads(_busca(DASH_COM_INDEX, timeout=60))
                except Exception as exc:
                    _diz(f"indice da comunidade indisponivel: {exc}")
            mudou = 0
            for item in CATALOGO:
                fonte = _fonte(item)
                if not fonte:
                    continue
                try:
                    if fonte == "dash":
                        versao, pacote = _versao_dash(item)
                    elif fonte == "dash_com":
                        versao, pacote = _versao_dash_com(item, indice)
                    elif fonte == "python":
                        versao, pacote = _versao_python(item)
                    elif fonte == "django":
                        versao, pacote = _versao_django(item)
                    else:
                        versao, pacote = _versao_git(item)
                except Exception as exc:
                    _diz(f"{item['nome']}: nao consegui conferir ({exc})")
                    continue
                if not versao:
                    # sem versao confiavel agora: apaga o que estava guardado, senao um
                    # valor velho e errado ("/9") fica na tela para sempre
                    achado.pop(item["id"], None)
                    continue
                antes = (achado.get(item["id"]) or {}).get("versao") or item.get("versao") or ""
                achado[item["id"]] = {"versao": versao, "pacote": pacote}
                if versao != antes:
                    mudou += 1
                    _diz(f"{item['nome']}: {antes or '?'} -> {versao}")
            achado["_quando"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _cache().write_text(json.dumps(achado, ensure_ascii=False, indent=1), encoding="utf-8")
            _atualizacao.update({"state": "done", "mudou": mudou})
            _diz(f"pronto: {mudou} versao(oes) mudaram" if mudou else "pronto: tudo ja estava atual")
        except Exception as exc:
            _atualizacao["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="catalogo-versoes", daemon=True).start()
    return {"accepted": True}


def listar(categoria: str = "") -> list[dict]:
    novas = versoes()
    itens = [i for i in CATALOGO if not categoria or i["categoria"] == categoria]
    saida = []
    for item in itens:
        atual = novas.get(item["id"]) or {}
        saida.append({**item,
                      "versao": atual.get("versao") or item.get("versao", ""),
                      "pacote": atual.get("pacote") or item["pacote"],
                      "tem_pacote": bool(atual.get("pacote") or item["pacote"]),
                      "categoria_nome": CATEGORIAS.get(item["categoria"], item["categoria"])})
    return saida


def por_id(item_id: str) -> dict | None:
    return next((i for i in CATALOGO if i["id"] == item_id), None)


def resumo() -> dict:
    itens = listar()
    return {"categorias": CATEGORIAS, "itens": itens,
            "com_pacote": sum(1 for i in itens if i["tem_pacote"]), "total": len(itens),
            "atualizado_em": versoes().get("_quando", ""),
            "atualizacao": {k: v for k, v in _atualizacao.items() if k != "linhas"}}
