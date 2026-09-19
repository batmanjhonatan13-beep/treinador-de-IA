# Treinador de IA

Ferramenta local para ensinar fatos a um modelo de linguagem e **provar que ele aprendeu**.
Você dá um arquivo (ou manda extrair um site de documentação), ela treina um LoRA, faz a
prova sem deixar o modelo consultar nada e só marca 🟢 quando ele acerta tudo de cabeça,
três vezes seguidas. No fim, empacota o conhecimento para rodar em outro servidor.

Roda na sua máquina. Sem API paga, sem mandar seus dados para fora.

## Instalar

```bash
./instalar.sh            # Linux e macOS: instala o Docker se faltar e sobe a página
```
```powershell
.\instalar.ps1           # Windows
```

Abre em `http://127.0.0.1:8765`. O script faz só isso: Docker + página no ar. **O resto
(modelo, Ollama, dependências de treino) você instala pela própria página, em Ambiente** —
aqui mesmo ou em outro servidor por SSH.

Daemon sem rede bridge (Docker rootless no WSL, por exemplo) cai sozinho para a rede do
host, pelo `docker-compose.hostnet.yml`.

## As seis páginas

| Página | O que faz |
|---|---|
| **Chat** | Conversa. Escolhe qual conhecimento consolidado o modelo usa e liga/desliga a consulta ao arquivo. |
| **Arquivos** | Manda `.md`, `.txt` ou `.jsonl`. Fora do formato, o modelo reescreve e você confere. Dá para apagar arquivo. |
| **Preparar dados** | Cola texto **ou extrai um site de documentação inteiro** e já devolve os fatos prontos. |
| **Treino** | Treina um arquivo (ou vários juntos), prova sem consulta e repete até consolidar. |
| **Exportar** | Empacota conhecimento consolidado + instalador do sistema escolhido. |
| **Ambiente** | Estado da conexão, catálogo de modelos e instalação local ou por SSH. |

Em toda página, a pílula no canto do menu diz **conectado** ou **não conectado**, com o que
está faltando.

## Como o conhecimento vira peso

```
arquivo .md ──► pares de treino ──► LoRA ──► prova sem consulta ──► 🟢 consolidado
                                     ▲                     │
                                     └──── errou? treina ──┘
```

- **Uma rodada** = `train.epochs` épocas sobre o arquivo. Tamanho fixo, para dar para comparar.
- **Uma tentativa** = um LoRA novo, do zero, treinado N rodadas, seguido de uma prova.
  N vem do histórico (rodadas por linha dos treinos que fecharam).
- **Consolidado** = **3 tentativas seguidas** acertando a prova inteira **já na primeira vez**
  (`learn.streak`). Uma tentativa isolada pode ser sorte de rodada; três do zero, não.
  Se uma falhar, a sequência zera e o ciclo treina aquela tentativa até fechar — assim
  descobre o N verdadeiro, que passa a valer para as próximas.
- Só conhecimento consolidado entra no chat e no export.
- **Parar** nunca corrompe: cada tentativa que passa é salva; só a em curso é descartada.
- **Apagar conhecimento** leva os pesos, os logs e o histórico daquele arquivo.

### Juntar conhecimentos

Não dá para somar dois LoRAs já treinados. Medido duas vezes: somar um sobre o outro vira
ruído (0/12); concatenar (o `cat` do PEFT) faz um responder pelo outro (3/11). Para dois
assuntos no mesmo peso, **selecione os dois arquivos no Treino** (Ctrl) — vira `a.md + b.md`,
um conhecimento só. Medido: 22/22 em 12 rodadas.

### O conhecimento é casado com o modelo

Um LoRA só funciona no modelo base em que foi treinado. Cada treino grava esse modelo; o
chat esconde o que não serve para o modelo atual, e o pacote exportado leva o modelo de
origem escrito no README e no `run.py`.

## Modelos

O catálogo (em `config.yaml`, visível na página Ambiente) diz o que cabe na máquina:

| Modelo | Tamanho | Treina sem GPU | RAM | VRAM |
|---|---|---|---|---|
| SmolLM2 135M / 360M | 135M / 360M | sim | 3 / 5 GB | — |
| Qwen2.5 0.5B / 1.5B | 0.5B / 1.5B | sim | 6 / 12 GB | 2 / 4 GB |
| Qwen2.5 3B | 3B | não | 24 GB | 8 GB |
| Qwen2.5 7B | 7B | não | 48 GB | 14 GB |

**Treino sem placa de vídeo funciona.** Medido: SmolLM2-360M com CUDA desligada chegou a
11/11 em 10 rodadas, 651 s. Com GPU, o Qwen 3B faz o mesmo arquivo em 35 s. Na CPU o modelo
carrega em float32 (bitsandbytes só quantiza em CUDA) e o lote da prova cai para 2.

## Extrair documentação

A raiz é só o começo: o extrator segue os links do domínio, aba por aba, e grava em disco
página por página. Usa o Chrome sem janela, então pega página montada por JavaScript;
descreve as imagens com `learn.vision_model`; e, com "seguir links para outras docs", entra
também nas páginas de outras ferramentas que a doc cita — um nível, sem sair rastejando a
web. Ao terminar, formata sozinho: você vê os fatos, não o texto bruto.

Dois filtros cuidam da qualidade:

- **Fato que não está no texto é barrado.** Numa página de instalação do Docker o modelo
  inventou `docker build` e `docker run`, que não estavam lá. A checagem compara o que
  sobrevive à tradução — comando entre crases, nome próprio, sigla, versão — porque a doc
  costuma estar em inglês e o fato sai em português.
- **Comando é conferido**: o que aparece em `código` ou bloco de código tem que virar fato;
  o que escapou ganha uma segunda passada.

## Preparar outra máquina

A página **Ambiente** olha o alvo (sistema, gerenciador de pacotes, root/sudo, Python,
compilador, GPU, RAM), monta o script conforme o que achou e mostra antes de rodar.

Sistemas: Ubuntu/Debian, **Amazon Linux 2023**, **Amazon Linux 2** (Python 3.8 pelos extras),
RHEL/Rocky/Alma/Fedora, **SUSE/openSUSE**, Arch, Alpine, macOS e Windows (winget/choco).

- **Instalar aqui** ou **no servidor (SSH)**. Por SSH é preciso chave, sem senha interativa;
  dá para enviar o código do projeto junto.
- **Sem root** ele ainda instala o Ollama na pasta do usuário.
- O **compilador C** entra no básico: sem ele o torch quebra ao compilar kernel na primeira
  passada de treino.
- Depois, "Usar o Ollama desse servidor" aponta o chat para a máquina preparada.

## Exportar

O pacote leva os pesos, um `run.py`, o README e o instalador com os pacotes certos do
sistema escolhido. No destino é um comando:

```bash
./instalar.sh
.venv/bin/python run.py "sua pergunta"
.venv/bin/python run.py --serve 8000     # HTTP: POST /chat {"pergunta": "..."}
```

**Carregue em 4 bits.** O adaptador foi treinado sobre a base quantizada; em precisão cheia
as mesmas perguntas saem diferentes (medido: 9/11 → 4/11). O `run.py` já faz isso quando há
GPU e avisa quando não dá.

## Decisões que custaram medição

- **LoRA na MLP**, não só na atenção. Fato declarativo mora ali; com `q/k/v/o` sozinhos o
  modelo não fixava um nome nem em 27 rodadas.
- **Prompt único** (`src/agentepc/prompting.py`) no treino, na prova e no chat. Prefixo
  diferente = peso existe mas não é acionado.
- **Sessão única** (`session.py`): o modelo fica na GPU do começo ao fim. Antes cada rodada
  pagava duas cargas do modelo (~90 s de carga para ~20 s de treino).
- **Perda só na resposta** (`completion_only_loss`) e nenhum par um-para-muitos no dataset.
- **Sem fusão**: fundir o LoRA no modelo e converter para GGUF derrubou a nota de 16/21 para
  8/21. O adaptador fica separado, como o treino o deixou.

## Onde ficam as coisas

Nada disso vai para o repositório — são dados da sua máquina.

```
data/conhecimento/   arquivos de treino          data/adapters/versions/  pesos por treino
data/extracoes/      sites extraídos + fatos     data/logs/               log por treino
data/provas/         prova de cada arquivo       data/treinos.jsonl       histórico
data/export/         pacotes prontos             data/skills/             regras de formatação
```

## Rodar à mão (sem Docker)

```bash
export PYTHONPATH=src
.venv-train/bin/python -m agentepc web        # http://127.0.0.1:8765
```

O treino precisa do `.venv-train` (torch, peft, trl). O chat sozinho roda no Ollama.
O container serve a página; o treino usa a máquina, porque precisa da GPU.
