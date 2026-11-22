#requires -Version 5.1
<#
.SYNOPSIS
    rag-agent 一键演示脚本：按顺序跑完「索引 → 入库 → 检索 → 回答 → 拒答 → Agent → 评测」。

.DESCRIPTION
    每一步都会先打印要执行的命令，再执行，方便现场边看边讲。
    默认使用 config.toml（DeepSeek + DashScope，需要 .env 里的 key）。
    加 -Offline 则使用 config.offline.toml（哈希向量 + 规则替身，零 Key、不联网）。

.EXAMPLE
    # 完整体验（真实模型）
    pwsh -File scripts\demo.ps1 -Pause

.EXAMPLE
    # 没有任何 API Key 时演示
    pwsh -File scripts\demo.ps1 -Offline -Pause

.EXAMPLE
    # 连评测一起跑（会做 max_unseen_ratio 阈值扫描，比较慢）
    pwsh -File scripts\demo.ps1 -FullEval
#>
[CmdletBinding()]
param(
    [switch]$Offline,
    [switch]$Pause,
    [switch]$FullEval,
    [switch]$SkipEval
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
chcp 65001 > $null

$Cli = Join-Path $Root ".venv\Scripts\ragagent.exe"
if (-not (Test-Path $Cli)) {
    Write-Host "找不到 .venv\Scripts\ragagent.exe" -ForegroundColor Red
    Write-Host "先执行：uv sync  或  pip install -e `".[dev]`"" -ForegroundColor Yellow
    exit 1
}

$ConfigArgs = @()
$ModeName = "在线（DeepSeek 对话 + text-embedding-v3 向量）"
$ReportPath = "reports/eval.md"
if ($Offline) {
    $ConfigArgs = @("--config", "config.offline.toml")
    $ModeName = "离线（哈希向量 + 规则替身，零 Key、不联网）"
    $ReportPath = "reports/eval-offline.md"
}

function Write-Act {
    param([int]$No, [string]$Title, [string]$Point)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host ("  第 {0} 幕  {1}" -f $No, $Title) -ForegroundColor Cyan
    Write-Host ("  讲点：{0}" -f $Point) -ForegroundColor DarkGray
    Write-Host ("=" * 70) -ForegroundColor DarkGray
}

function Invoke-Demo {
    param([string[]]$Arguments)
    $shown = if ($ConfigArgs.Count -gt 0) { "--config config.offline.toml " + ($Arguments -join " ") } else { $Arguments -join " " }
    Write-Host ("`$ ragagent {0}" -f $shown) -ForegroundColor Yellow
    $all = @()
    if ($ConfigArgs.Count -gt 0) { $all += $ConfigArgs }
    $all += $Arguments
    & $Cli @all
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ↑ 返回码非 0：在线模式请检查 .env 里的 key 和网络" -ForegroundColor Red
    }
    if ($Pause) { Read-Host "`n按回车继续" | Out-Null }
}

Write-Host ""
Write-Host "  rag-agent 演示" -ForegroundColor Green
Write-Host "  模式：$ModeName"
Write-Host ("  配置：{0}" -f $(if ($Offline) { "config.offline.toml" } else { "config.toml" }))
Write-Host ("  目录：{0}" -f $Root)

Write-Act 1 "索引现状" "先让人看到数据在哪、用的是哪套向量"
Invoke-Demo @("stats")

Write-Act 2 "入库（幂等）" "同一份文档重复入库会 skipped，靠 sha256 判断，不重复计费"
Invoke-Demo @("ingest", "data/docs")

Write-Act 3 "检索（可解释）" "每条命中都带 bm25 / vector / fused / rerank 四层分数和 reasons"
Invoke-Demo @("search", "退货运费谁承担", "-k", "3")

Write-Act 4 "带引用回答" "mode=llm 说明真模型生效；citations 指向真实 chunk_id"
Invoke-Demo @("ask", "退货运费谁承担")

Write-Act 5 "拒答" "知识库里没有的问题不硬答，refused=true 是正确行为"
Invoke-Demo @("ask", "支持以旧换新吗")

Write-Act 6 "Agent 多步" "工具调用 + 每步 trace + 提交时校验引用"
Invoke-Demo @("agent", "退货时效和换货时效有什么区别")

if (-not $SkipEval) {
    Write-Act 7 "离线评测" "指标可复现，报告里有 max_unseen_ratio 的取舍表"
    $evalArgs = @("eval", "--report", $ReportPath)
    if (-not $FullEval) { $evalArgs += "--no-sweep" }
    Invoke-Demo $evalArgs
    Write-Host ""
    Write-Host ("  报告已写入：{0}" -f $ReportPath) -ForegroundColor Green
    if (-not $FullEval) {
        Write-Host "  想要阈值扫描表（recall vs 误拒 vs 拒答准确率）：离线模式再加 -FullEval（在线语义校验模式会自动跳过）" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor DarkGray
Write-Host "  演示结束。HTTP 版：" -ForegroundColor Green
Write-Host "    ragagent serve --port 8000   然后打开 http://127.0.0.1:8000/（网页版，/docs 是 OpenAPI）" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor DarkGray
