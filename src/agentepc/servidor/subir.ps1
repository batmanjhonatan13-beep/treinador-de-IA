# Sobe o roteador deste pacote no Windows.
#   .\subir.ps1                 porta 8770
#   .\subir.ps1 --porta 9000
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Write-Output "== primeira vez: preparando o ambiente"
  if (-not (Get-Command python -EA 0)) {
    if (Get-Command winget -EA 0) {
      winget install --silent --accept-package-agreements --accept-source-agreements --id Python.Python.3.12
    } else { throw "instale o Python 3 e rode de novo" }
  }
  python -m venv .venv
  .\.venv\Scripts\pip.exe install --upgrade pip
  if (Test-Path .\requirements.txt) { .\.venv\Scripts\pip.exe install -r .\requirements.txt }
}

$temGpu = & .\.venv\Scripts\python.exe -c "import torch;print(torch.cuda.is_available())" 2>$null
if ($temGpu -eq "True") { Write-Output "== placa de video encontrada" }
else { Write-Output "== sem placa de video: imagem e 3D nao rodam aqui" }

& .\.venv\Scripts\python.exe roteador.py $args
