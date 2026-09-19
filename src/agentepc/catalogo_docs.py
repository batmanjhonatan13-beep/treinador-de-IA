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


def listar(categoria: str = "") -> list[dict]:
    itens = [i for i in CATALOGO if not categoria or i["categoria"] == categoria]
    return [{**i, "tem_pacote": bool(i["pacote"]),
             "categoria_nome": CATEGORIAS.get(i["categoria"], i["categoria"])} for i in itens]


def por_id(item_id: str) -> dict | None:
    return next((i for i in CATALOGO if i["id"] == item_id), None)


def resumo() -> dict:
    return {"categorias": CATEGORIAS, "itens": listar(),
            "com_pacote": sum(1 for i in CATALOGO if i["pacote"]), "total": len(CATALOGO)}
