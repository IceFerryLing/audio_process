param(
    [string]$Config = "configs/data/librispeech_stage2.yaml",
    [ValidateRange(1, 16)]
    [int]$Chunks = 8,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
# 通过已校验的YAML契约统一解析URL、大小、哈希和目标路径。
Push-Location $repoRoot
try {
    $specText = (& sylrec data mfa-download-spec --config $Config) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve MFA download specification."
    }
    $spec = $specText | ConvertFrom-Json
} finally {
    Pop-Location
}

$destination = [IO.Path]::GetFullPath($spec.destination)
$parallelDestination = "$destination.parallel"
$partRoot = [IO.Path]::GetFullPath("$destination.parts")
$workspacePrefix = $repoRoot.TrimEnd('\') + '\'
# 脚本只能在当前仓库工作区内创建和移动文件。
if (-not $destination.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Download destination is outside the workspace."
}
if (-not $partRoot.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Part directory is outside the workspace."
}
# 只有发布文件大小和SHA-256同时匹配时，重跑才允许跳过。
if (
    (Test-Path $destination) -and
    (Get-Item $destination).Length -eq $spec.size_bytes -and
    (Get-FileHash $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $spec.sha256 -and
    -not $Overwrite
) {
    Write-Output "{`"event`":`"mfa_model_download`",`"status`":`"skipped`",`"bytes`":$($spec.size_bytes)}"
    exit 0
}
if ((Test-Path $destination) -and -not $Overwrite) {
    throw "Existing model archive is incomplete; rerun with -Overwrite explicitly."
}

New-Item -ItemType Directory -Force -Path $partRoot | Out-Null
$chunkSize = [math]::Ceiling($spec.size_bytes / $Chunks)
$jobs = @()
$completedParts = @()
# 将发布文件切成确定的字节区间，使已完成的有效分片可以续传复用。
for ($index = 0; $index -lt $Chunks; $index++) {
    $start = [int64]($index * $chunkSize)
    $end = [int64][math]::Min($spec.size_bytes - 1, (($index + 1) * $chunkSize) - 1)
    if ($start -gt $end) { break }
    $partPath = Join-Path $partRoot ("{0:D3}.part" -f $index)
    $expectedBytes = $end - $start + 1
    if ((Test-Path $partPath) -and (Get-Item $partPath).Length -eq $expectedBytes) {
        $completedParts += [pscustomobject]@{
            Path = $partPath
            Start = $start
            End = $end
            Bytes = $expectedBytes
        }
        continue
    }
    # PowerShell job并行下载独立区间，curl负责瞬时网络错误重试。
    $jobs += Start-Job -ScriptBlock {
        param($Url, $Start, $End, $Path)
        & curl.exe --silent --show-error --ssl-no-revoke -fL `
            --retry 5 --retry-all-errors --retry-delay 2 `
            --speed-limit 1024 --speed-time 30 `
            --range "$Start-$End" -o $Path $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl range $Start-$End failed with exit code $LASTEXITCODE"
        }
        [pscustomobject]@{Path=$Path; Start=$Start; End=$End; Bytes=(Get-Item $Path).Length}
    } -ArgumentList $spec.url, $start, $end, $partPath
}

$completedSuccessfully = $false
try {
    if ($jobs.Count -gt 0) {
        $jobs | Wait-Job | Out-Null
    }
    $failed = @($jobs | Where-Object {$_.State -ne "Completed"})
    if ($failed.Count -gt 0) {
        $failed | Receive-Job
        throw "$($failed.Count) ranged downloads failed."
    }
    $downloadedParts = @($jobs | Receive-Job)
    $parts = @(($completedParts + $downloadedParts) | Sort-Object Start)
    foreach ($part in $parts) {
        $expected = $part.End - $part.Start + 1
        if ($part.Bytes -ne $expected) {
            throw "Range $($part.Start)-$($part.End) has $($part.Bytes) bytes; expected $expected."
        }
    }

    # 先合并到临时文件，避免中断产物被误认为完整模型包。
    $output = [IO.File]::Open($parallelDestination, [IO.FileMode]::Create, [IO.FileAccess]::Write)
    try {
        foreach ($part in $parts) {
            $input = [IO.File]::OpenRead($part.Path)
            try { $input.CopyTo($output) } finally { $input.Dispose() }
        }
    } finally {
        $output.Dispose()
    }
    if ((Get-Item $parallelDestination).Length -ne $spec.size_bytes) {
        throw "Combined archive size does not match release metadata."
    }
    Move-Item -LiteralPath $parallelDestination -Destination $destination -Force
    # 所有分片合并后，以官方SHA-256作为最终验收门禁。
    $hash = (Get-FileHash $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $spec.sha256) {
        throw "Combined archive hash does not match the pinned model."
    }
    $completedSuccessfully = $true
    Write-Output "{`"event`":`"mfa_model_download`",`"status`":`"downloaded`",`"bytes`":$($spec.size_bytes),`"sha256`":`"$hash`"}"
} finally {
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
    # 失败时保留分片以便续传，只有完整校验通过后才清理。
    if ($completedSuccessfully -and (Test-Path $partRoot)) {
        Remove-Item -LiteralPath $partRoot -Recurse -Force
    }
}
