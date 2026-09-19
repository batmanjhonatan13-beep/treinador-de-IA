# Instala o Docker Desktop (se faltar) e sobe a pagina do agente-pc no Windows.
# Rode no PowerShell:  .\instalar.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$porta = if ($env:PORTA) { $env:PORTA } else { "8765" }

function Diga($m) { Write-Output "`n== $m" }

Diga "verificando o Docker"
if (Get-Command docker -ErrorAction SilentlyContinue) {
  docker --version
} else {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --silent --accept-package-agreements --accept-source-agreements --id Docker.DockerDesktop
    Write-Output "abra o Docker Desktop uma vez e rode este script de novo"
    exit 1
  }
  throw "instale o Docker Desktop (https://www.docker.com/products/docker-desktop/) e rode de novo"
}

Diga "subindo a pagina"
docker compose up -d --build

Write-Output ""
Write-Output "Pronto: http://127.0.0.1:$porta"
Write-Output "Na pagina, use 'Preparar ambiente' para instalar o modelo e o treino aqui ou em outro servidor."
Write-Output "Parar: docker compose down"
