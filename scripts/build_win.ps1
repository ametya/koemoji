# Windows 用配布ビルドスクリプト（CPU前提・追加環境不要の zip を作る）
# 使い方:  powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1
#requires -Version 5.1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    Write-Host "== 1/4 ビルド用 venv を準備"
    if (-not (Test-Path ".venv-build\Scripts\python.exe")) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            & py -3.13 -m venv .venv-build 2>$null
            if (-not (Test-Path ".venv-build\Scripts\python.exe")) {
                & py -3.12 -m venv .venv-build
            }
        } else {
            python -m venv .venv-build
        }
    }
    $python = Join-Path $root ".venv-build\Scripts\python.exe"

    Write-Host "== 2/4 依存関係をインストール"
    & $python -m pip install --upgrade pip --quiet
    & $python -m pip install ".[build]" --quiet

    Write-Host "== 3/4 PyInstaller でビルド"
    & $python -m PyInstaller packaging\koemoji.spec --noconfirm --clean --distpath dist --workpath build

    Write-Host "== 4/4 zip にまとめる"
    Copy-Item README.md (Join-Path "dist\koemoji" "README.md") -Force
    Compress-Archive -Path "dist\koemoji" -DestinationPath "dist\koemoji-windows-x64.zip" -Force

    Write-Host ""
    Write-Host "ビルド完了: dist\koemoji-windows-x64.zip"
    Write-Host "  dist\koemoji\koemoji-gui.exe … GUI（ダブルクリックで起動）"
    Write-Host "  dist\koemoji\koemoji.exe     … CLI"
} finally {
    Pop-Location
}
