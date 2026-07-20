param(
    [string]$Config = "configs/data/librispeech_stage2.yaml",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
# 由Python统一校验YAML，并返回一份机器可读的运行规范。
Push-Location $repoRoot
try {
    $specText = (& sylrec data mfa-spec --config $Config) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve MFA runtime specification."
    }
    $spec = $specText | ConvertFrom-Json
} finally {
    Pop-Location
}

$corpus = $spec.corpus_directory
$dictionary = $spec.dictionary_path
$modelArchive = $spec.acoustic_model_archive
$aligned = $spec.textgrid_directory

function Convert-ToWslPath([string]$Path) {
    # MFA运行在WSL内，因此显式把C:\path转换为/mnt/c/path，避免shell自行猜测。
    $forward = $Path.Replace('\', '/')
    if ($forward -notmatch '^([A-Za-z]):/(.*)$') {
        throw "Cannot convert path to WSL form: $Path"
    }
    $drive = $Matches[1].ToLowerInvariant()
    return "/mnt/$drive/$($Matches[2])"
}

$corpusWsl = Convert-ToWslPath $corpus
$dictionaryWsl = Convert-ToWslPath $dictionary
$modelArchiveWsl = Convert-ToWslPath $modelArchive
$alignedWsl = Convert-ToWslPath $aligned
$mfa = $spec.mfa_runner

# TextGrid数量必须精确匹配才可幂等跳过，不完整输出必须显式处理。
$existing = @(Get-ChildItem $aligned -Recurse -Filter *.TextGrid -ErrorAction SilentlyContinue)
if ($existing.Count -eq $spec.expected_textgrids -and -not $Overwrite) {
    Write-Output "{`"event`":`"mfa_align`",`"status`":`"skipped`",`"textgrids`":$($spec.expected_textgrids)}"
    exit 0
}
if ($existing.Count -gt 0 -and -not $Overwrite) {
    throw "Existing MFA output is incomplete; rerun with -Overwrite explicitly."
}

# corpus、词典或模型校验失败时，禁止继续执行对齐。
wsl -d ubuntu -- bash -lc "$mfa validate '$corpusWsl' '$dictionaryWsl' '$modelArchiveWsl' --clean"
if ($LASTEXITCODE -ne 0) {
    throw "MFA corpus validation failed with exit code $LASTEXITCODE."
}
# Long TextGrid保留data/alignment.py需要解析的word和phone层。
wsl -d ubuntu -- bash -lc "$mfa align '$corpusWsl' '$dictionaryWsl' '$modelArchiveWsl' '$alignedWsl' --output_format '$($spec.output_format)' --clean"
if ($LASTEXITCODE -ne 0) {
    throw "MFA alignment failed with exit code $LASTEXITCODE."
}
wsl -d ubuntu -- bash -lc "$mfa version"
if ($LASTEXITCODE -ne 0) {
    throw "MFA version check failed with exit code $LASTEXITCODE."
}
