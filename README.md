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

### Quando o Docker "some"

Quase sempre não sumiu. Os três casos, e o que cada um quer dizer:

| O que aparece | O que é | Como resolver |
|---|---|---|
| `command not found` com `sudo` | o binário está em `~/.local/bin`, e o `sudo` não usa esse caminho | rode sem `sudo`, ou use o Docker do sistema |
| `cannot connect ... docker.sock: no such file` | o daemon está parado (o **rootless não volta sozinho** depois de reiniciar) | `./docker/subir-docker.sh` |
| `permission denied ... docker.sock` | o daemon está de pé, mas seu usuário não está no grupo `docker` | `sudo usermod -aG docker $USER` e abra um terminal novo (no Windows: `wsl --shutdown`) |

Dá para ter **dois Dockers** na mesma máquina — o Desktop (socket do sistema) e um rootless
na pasta do usuário. O `./docker/subir-docker.sh` procura um que responda e diz qual usar; a
página **Ambiente** mostra o mesmo, escrito. Com o Desktop funcionando, o rootless vira
sobra: `pkill -f dockerd-rootless` e `rm -rf ~/.local/bin/docker*` limpam.

## As dez páginas

| Página | O que faz |
|---|---|
| **Chat** | Conversa. Escolhe qual conhecimento consolidado o modelo usa e liga/desliga a consulta ao arquivo. |
| **Arquivos** | Manda `.md`, `.txt` ou `.jsonl`. Fora do formato, o modelo reescreve e você confere. Dá para apagar arquivo. |
| **Preparar dados** | Cola texto **ou extrai um site de documentação inteiro** e já devolve os fatos prontos. |
| **Treino** | Treina um arquivo (ou vários juntos), prova sem consulta e repete até consolidar. |
| **Imagens** | Junta imagens com licença declarada e treina um **LoRA de estilo** no Stable Diffusion. |
| **Classificador** | Separa categorias suas em imagens, com prova em imagens que ele nunca viu. |
| **Som** | Junta áudio com licença, **classifica som** (com prova), **gera som** e ensina um estilo ao gerador. |
| **3D** | Texto ou foto → malha `.glb`, e referências CC0 do Poly Haven. |
| **Exportar** | Empacota conhecimento consolidado + instalador do sistema escolhido. |
| **Ambiente** | Estado da conexão, **catálogo de treinos com instalação sob demanda**, modelos e instalação por SSH. |

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

### Trocar de modelo

"Usar este" na página Ambiente **desinstala o anterior e instala o escolhido**: tira o modelo
do Ollama, apaga os pesos de treino do cache (e os blobs que ficarem sem dono) e apaga o
conhecimento treinado que só servia nele. A confirmação lista exatamente o que sai. Depois
baixa o modelo novo e os pesos de treino dele.

Há também **Limpar pesos órfãos do cache**, para o que sobrou de trocas anteriores.

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

Em três tempos, para você não precisar adivinhar um número de páginas:

1. **Procurar páginas** — anda a árvore a partir da raiz e lista cada página **pelo nome**.
   Termina com um aviso claro (“BUSCA CONCLUÍDA — N páginas”), dizendo quantos links ficaram
   de fora pelo teto.
2. **Marcar e extrair** — você escolhe quais entram e elas viram um **lote** com nome (o
   assunto que você deu; sem ele, um nome tirado da URL).
3. **Formatar o lote** — um de cada vez. Enquanto um lote formata, nova busca é recusada, e
   dar F5 continua mostrando “formatando” com o botão de parar. Terminou, o texto bruto é
   apagado e fica só o arquivo de fatos. Lotes velhos podem ser apagados na lista.

Usa o Chrome sem janela, então pega página montada por JavaScript; descreve as imagens com
`learn.vision_model`; e, com "seguir links para outras docs", entra também nas páginas de
outras ferramentas que a doc cita — um nível, sem sair rastejando a web.

### Quando o site corta o acesso

Andar página a página por uma documentação grande faz o servidor barrar: ele passa a
devolver a tela de **“Um momento…”** (verificação de robô) com HTTP 200, e um extrator
ingênuo conta aquilo como página lida. Aconteceu aqui com a doc da Godot: **2 273 páginas
lidas, 114 aproveitadas** — o resto era a mesma tela de verificação.

Agora o extrator reconhece essa tela, **não conta como página lida**, e para depois de três
seguidas dizendo o que fazer. O mesmo vale para o `429` (“vá mais devagar”).

O caminho certo para doc grande é o pacote que o próprio site publica:

```
https://docs.exemplo.org/_/downloads/<idioma>/<versão>/htmlzip/
```

O botão **Importar documentação (.zip)** baixa isso uma vez só (ou lê um `.zip` do seu
disco), lista as páginas igual a uma busca, e você marca o que quer. Um download em vez de
mil requisições — e nada de verificação de robô.

### Filtros de endereço

O endereço é que diz o assunto, então é por ele que se filtra:

- **Só abaixo do endereço raiz** (ligado por padrão): a raiz `docs.python.org/pt-br/3/` não
  deixa entrar `/pt-br/3.13/` nem `/es/3/`. Resolve o caso de doc com várias versões.
- **Só endereços contendo**: `/spaces/TIME/` pega um espaço do Confluence e ignora o resto.
- **Pular endereços contendo**: `/blog/`, `/es/`, o que atrapalhar.

Dois filtros cuidam da qualidade:

- **Fato que não está no texto é barrado.** Numa página de instalação do Docker o modelo
  inventou `docker build` e `docker run`, que não estavam lá. A checagem compara o que
  sobrevive à tradução — comando entre crases, nome próprio, sigla, versão — porque a doc
  costuma estar em inglês e o fato sai em português.
- **Comando é conferido**: o que aparece em `código` ou bloco de código tem que virar fato;
  o que escapou ganha uma segunda passada.
- **O trecho nunca atravessa página nem seção.** Antes o corte era só por tamanho, e a
  explicação de um módulo podia ser partida no meio, com metade indo para outro lote sem o
  contexto. Agora cada título fecha o trecho.
- **O fato nasce com o contexto.** O trecho viaja junto com a trilha de títulos. Assim o item “Remove os parâmetros type, choices e metavar” vira
  “No Python 3.14 foram removidos os parâmetros `type`, `choices` e `metavar` de
  `argparse.BooleanOptionalAction`, descontinuados desde o Python 3.12”.
- **Fato longo fica longo.** Comando e explicação técnica são compridos; cortar em pedaços
  curtos estraga o sentido. O Conferir só aponta linha curta demais, primeira pessoa,
  repetição e segredo — e o botão **Consertar** arruma isso.

## O que dá para treinar

O catálogo fica em **Ambiente** e se libera sozinho conforme a máquina: cada treino declara
VRAM, RAM, pacotes e programas, e o que não passa aparece bloqueado **com o motivo escrito**.
Nada é instalado sem você pedir — cada treino tem seu botão de instalar.

| Treino | Nesta máquina (RTX 3060) | Consolidação |
|---|---|---|
| Fatos em texto (LoRA) | ✅ roda até sem GPU | 3 tentativas limpas — automática |
| Estilo de imagem (SD 1.5) | ✅ 3 min / 200 passos | você aprova as amostras |
| Personagem ou objeto | ✅ mesma base, legenda com gatilho | você aprova |
| Estilo em SDXL | ✅ pesado: 30 passos ≈ 5 min | você aprova |
| Classificador de imagem | ✅ segundos | acerto em imagens nunca vistas — automática |
| Classificador de som | ✅ segundos, roda sem GPU | acerto em faixas nunca ouvidas — automática |
| Gerar som e música (MusicGen) | ✅ 5 s de áudio em ~40 s | você ouve e aprova |
| Estilo de som (LoRA no MusicGen) | ✅ precisa de ~20 faixas para valer | você ouve e aprova |
| Texto ou foto → 3D (Shap-E) | ✅ 11 s por malha, sai `.glb`/`.obj` | você olha e aprova |
| Fotos → 3D (Gaussian Splatting) | 🕓 falta COLMAP (precisa de root) | — |
| Clonar voz | 🕓 não implementado; cuidado com a licença | — |
| Treinar vídeo | ✕ cluster, não PC | — |
| Gerador 3D do zero (tipo Meshy) | ✕ ~10 TB de dados e GPU de datacenter | — |

**A consolidação muda com o tipo.** Onde existe resposta certa (texto, classificador) a
máquina decide sozinha. Onde não existe (estilo, personagem) **você** decide, no chat,
olhando o que saiu — e a sua decisão fica gravada com data. Inventar um número ali seria
mentira; deixar tudo "talvez" seria inútil.

Só o que está consolidado aparece no **Exportar**, e cada tipo sai com o seu próprio README
de uso (LoRA para Automatic1111/ComfyUI, classificador com o código de carregar, malha 3D
pronta para o Blender).

### Duas coisas que não são treino, e está escrito na tela

**3D não se treina em casa.** O que a página 3D faz é rodar o Shap-E (OpenAI, Apache-2.0),
que já vem no `diffusers`. Escolhi ele e não o TripoSR — que gera malha melhor — porque o
TripoSR compila código CUDA na instalação (`torchmcubes`) e quebra em máquina sem compilador
C: um cliente seu não conseguiria instalar. O que dá para treinar é o lado da **imagem**: um
LoRA faz o seu objeto sair sempre igual, e essa imagem vira malha.

**Gerar som também é modelo pronto** (MusicGen, da Meta). O que se treina ali é um LoRA por
cima dele. Repare na licença: o MusicGen é **CC-BY-NC** — serve para uso interno, mas o áudio
gerado não pode ser vendido. Para uma ferramenta que vai ser vendida, a licença do modelo
pesa tanto quanto a qualidade.

### De onde vem o material

| Fonte | O que traz | Licença |
|---|---|---|
| Openverse | imagens e **áudio** (indexa o Freesound) | filtro por licença na busca |
| Wikimedia Commons | imagens e áudio | declarada em cada arquivo |
| Poly Haven | modelos 3D | tudo CC0 |

Cada download grava autor, licença e endereço em `creditos.json`. **Spotify, YouTube, Deezer
e Google Imagens ficam de fora de propósito**: os termos de uso deles proíbem baixar e usar
para treino, e o arquivo baixado não diz quem é o dono. Isso não muda por a ferramenta rodar
na sua máquina — muda quando você for vender o que saiu dela.

## Testar no chat

O Chat tem duas abas: **Texto** conversa com o modelo; **Imagem** gera com um estilo
treinado, com controle de força, e traz os botões *O estilo pegou* / *Não pegou* — é essa
resposta que consolida o treino de imagem.

## Treinar um estilo de imagem

Mesma técnica (LoRA), outro tipo de modelo. **Só com GPU** — difusão na CPU não é lenta, é
inviável. Medido nesta máquina (RTX 3060): SmolLM2 de texto treina na memória; o estilo de
imagem não.

1. **Juntar imagens** — busca no Openverse e no Wikimedia Commons, que declaram a licença.
   Autor, licença e endereço de cada arquivo ficam em `creditos.json`. Para usar arte sua,
   basta colocar os arquivos numa pasta dentro de `data/imagens/`.
2. **Treinar** — LoRA só na atenção do UNet, que é onde o estilo pega. Medido: 12 imagens,
   200 passos, 512 px → **3 minutos**, adaptador de 3,2 M pesos.
3. **Julgar** — ele gera amostras e você decide. **Não existe selo verde aqui**: estilo não
   tem resposta certa, então um número de "acertou" seria inventado.

O `.safetensors` que sai funciona no Automatic1111, ComfyUI ou em qualquer lugar que aceite
LoRA de fora.

**Sobre as imagens que você junta:** o coletor só usa fontes que dizem a licença, e guarda o
crédito. Raspar arte de Pinterest, ArtStation ou Instagram para copiar o traço de alguém é
outra conversa — o arquivo baixado não diz nada sobre o direito de uso, e publicar o
resultado pode dar problema. A ferramenta não faz isso.

## Preparar outra máquina

A página **Ambiente** olha o alvo (sistema, gerenciador de pacotes, root/sudo, Python,
compilador, GPU, RAM), monta o script conforme o que achou e mostra antes de rodar.

Sistemas: Ubuntu/Debian, **Amazon Linux 2023**, **Amazon Linux 2** (Python 3.8 pelos extras),
RHEL/Rocky/Alma/Fedora, **SUSE/openSUSE**, Arch, Alpine, macOS e Windows (winget/choco).

- **Instalar aqui** ou **no servidor (SSH)**. Por SSH é preciso chave (sem senha interativa);
  dá para enviar o código do projeto junto.
- Em *O que instalar* dá para marcar **quais treinos** vão junto (imagem, som, 3D…). O script
  leva os nomes dos pacotes no corpo, então funciona mesmo em servidor que não tem o projeto.
- **Senha de administrador**: quando a máquina exige sudo, o campo é obrigatório — sem ele a
  instalação nem começa. A senha vai pela entrada do shell (não fica em arquivo nem na lista
  de processos) e não aparece no script nem no log.
- Cada etapa diz **OK** ou **FALHOU** e o trabalho continua; no fim o resumo aponta o que
  falhou e como resolver.
- **Sem root** ele ainda instala o Ollama na pasta do usuário.
- O **compilador C** entra no básico: sem ele o torch quebra ao compilar kernel na primeira
  passada de treino.
- Depois, "Usar o Ollama desse servidor" aponta o chat para a máquina preparada.

## Exportar

### O pacote é um servidor: um endereço, vários modelos

Os cinco treinos **não viram um modelo só** — são arquiteturas diferentes, cada uma em cima
da sua base. O que o pacote leva é um **roteador** na frente delas:

```bash
./subir.sh                  # porta 8770
./subir.sh --manter 0       # nada dorme (mais rápido, come memória)
./subir.sh --acordar texto  # já sobe com o texto ligado
```

| Chamada | O que acontece |
|---|---|
| `GET /v1/models` | lista as peças do pacote |
| `POST /v1/chat/completions` | formato da OpenAI — **qualquer ferramenta que fale com a OpenAI fala com isto** |
| `POST /v1/images/generations` | PNG em base64 |
| `POST /v1/audio/generations` | WAV em base64 |
| `POST /v1/3d/generations` | `.glb` em base64 |
| `POST /classificar` | categoria + certeza de cada uma |
| `GET /status` · `POST /ligar` · `POST /desligar` | quem está ligado, e controle na mão |

**Ligar e desligar sozinho:** a peça só carrega no primeiro pedido dela e dorme depois de
`--manter` segundos parada. **Uma peça de GPU por vez** — antes de acordar uma, as outras
saem da placa; numa máquina com uma placa só não cabe tudo junto, e é melhor o roteador
decidir isso do que a sua ferramenta descobrir com um erro de memória.

| Tipo | GPU | CPU |
|---|---|---|
| texto | recomendado (6 GB) | funciona, devagar |
| imagem | **obrigatório** (6 GB) | não roda — difusão em CPU é inviável, não lenta |
| som (estilo) | recomendado (4 GB) | funciona, muito devagar |
| som (classificador) | opcional | funciona bem |
| classificador de imagem | opcional | funciona bem |
| 3D | **obrigatório** (6 GB) | não roda |

Cada pacote sai com um **`API.md`** documentando cada chamada, com exemplo em `curl` e em
Python — sem isso, quem recebe o pacote tem os modelos mas não sabe como pedir nada.

O roteador **não tem senha**. Rede interna, ou um proxy com autenticação na frente.

### Vários conhecimentos num pacote só

Marque tudo que vai para a mesma máquina — a doc que virou peso, o estilo de imagem, o som,
o classificador — e sai **uma pasta** assim:

```
pacote/
  README.md          o que tem dentro e em cima de que modelo cada peça roda
  instalar.sh        prepara tudo de uma vez, no sistema que você escolheu
  texto/<nome>/      adaptador + run.py
  imagem/<nome>/     LoRA .safetensors
  som-estilo/<nome>/ LoRA + usar.py que gera áudio
  som-classificador/<nome>/  modelo.pt + usar.py que roda sozinho
  3d/<nome>/         .glb, .obj, .ply e a prévia
```

**Isto não é fusão.** Dois conhecimentos de *texto* no mesmo pacote continuam carregando um
de cada vez: juntar adaptadores já treinados foi medido aqui e piora o resultado (16/21 caiu
para 8/21). Para os dois assuntos numa resposta só, treine os arquivos **juntos** na página
Treino. O README do pacote diz isso a quem receber, para ninguém tentar de novo.

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
