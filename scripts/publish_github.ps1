# GitHub リポジトリ作成 & push スクリプト
# 使い方:  powershell -ExecutionPolicy Bypass -File scripts\publish_github.ps1
# gh (GitHub CLI) がなければ winget でインストールし、ログイン後に
# このフォルダをそのままリポジトリとして公開します。
#requires -Version 5.1

param(
    [string]$RepoName = "koemoji",
    [string]$Description = "CPUだけで動くオフライン日本語文字起こしツール（faster-whisper / Windows GUI+CLI）",
    [switch]$Private
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-Host "GitHub CLI (gh) が見つかりません。winget でインストールします..."
        winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements
        $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$machinePath;$userPath"
    }

    gh auth status 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GitHub にログインします（ブラウザが開きます）..."
        gh auth login
    }

    $visibility = if ($Private) { "--private" } else { "--public" }
    gh repo create $RepoName $visibility --source . --remote origin --push --description $Description
    Write-Host ""
    Write-Host "完了: https://github.com/$((gh api user --jq .login))/$RepoName"
    Write-Host "リリース配布物を作るには: git tag v0.1.0; git push origin v0.1.0"
} finally {
    Pop-Location
}
