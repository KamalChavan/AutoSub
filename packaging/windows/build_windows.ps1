<#
.SYNOPSIS
  Build standalone Windows x64 AutoSub portable ZIP and Inno Setup (.exe) installer.

.DESCRIPTION
  Bundles:
    - Python runtime + PyYAML + onnxruntime + numpy + certifi (zero PyTorch)
    - Silero VAD ONNX model (`models/silero_vad.onnx`)
    - Windows x64 `ffmpeg.exe` (`bin/ffmpeg.exe`):
        * Uses `-FfmpegPath` if specified
        * Otherwise prefers `build/cache/ffmpeg-minimal.exe` (~4 MB audio-only build from `build_minimal_ffmpeg.sh`)
        * Otherwise downloads static `ffmpeg.exe` from `-FfmpegUrl` (unless `-NoBundleFfmpeg` is set)
    - Editable `config/`, `profiles/`, `glossaries/`, `.env.example`, `README.md`

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1 -FfmpegPath build\cache\ffmpeg-minimal.exe
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1 -NoBundleFfmpeg
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$FfmpegPath = "",
    [string]$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    [switch]$NoBundleFfmpeg,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$Description = ""
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        $label = if ($Description) { $Description } else { "$FilePath $($Arguments -join ' ')" }
        throw "Command failed with exit code $LASTEXITCODE`: $label"
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Push-Location $ProjectRoot

try {
    # 1. Read version from pyproject.toml
    $PyProject = Get-Content (Join-Path $ProjectRoot "pyproject.toml") -Raw
    if ($PyProject -match 'version\s*=\s*"([^"]+)"') {
        $AppVersion = $Matches[1]
    } else {
        $AppVersion = "0.1.0"
    }
    Write-Host "=== Building AutoSub v$AppVersion (Windows x64) ===" -ForegroundColor Cyan

    # 2. Prepare isolated build virtual environment
    $VenvDir = Join-Path $ProjectRoot "build\win-venv"
    if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
        Write-Host "[1/6] Creating isolated build virtualenv..."
        Invoke-Checked $PythonExe @("-m", "venv", $VenvDir) "create build venv"
    }
    $VenvPy = Join-Path $VenvDir "Scripts\python.exe"

    Write-Host "[2/6] Installing runtime + VAD (ONNX C library only, zero NumPy/PyTorch) + PyInstaller..."
    Invoke-Checked $VenvPy @("-m", "pip", "install", "--upgrade", "pip") "upgrade pip"
    Invoke-Checked $VenvPy @("-m", "pip", "install", "PyYAML>=6.0", "certifi>=2024.0", "pyinstaller>=6.10") "install runtime deps"
    Invoke-Checked $VenvPy @("-m", "pip", "install", "--no-deps", "onnxruntime>=1.18", "silero-vad") "install onnxruntime & silero-vad (--no-deps)"

    # Run offline unit tests before packaging (strictly fail build if any test fails)
    Write-Host "[2.5/6] Running offline unit tests..."
    Invoke-Checked $VenvPy @("-m", "unittest", "discover", "-s", "tests", "-v") "offline unit tests"

    # 3. Resolve ffmpeg.exe (unless -NoBundleFfmpeg)
    $CacheDir = Join-Path $ProjectRoot "build\cache"
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $MinimalFfmpeg = Join-Path $CacheDir "ffmpeg-minimal.exe"
    $CachedFfmpeg = Join-Path $CacheDir "ffmpeg.exe"
    $SelectedFfmpeg = ""

    if ($NoBundleFfmpeg) {
        Write-Host "[3/6] Skipping FFmpeg bundling (-NoBundleFfmpeg; target machine must provide ffmpeg in PATH)."
    } elseif ($FfmpegPath -and (Test-Path $FfmpegPath)) {
        $SelectedFfmpeg = (Resolve-Path $FfmpegPath).Path
    } elseif (Test-Path $MinimalFfmpeg) {
        $SelectedFfmpeg = $MinimalFfmpeg
        Write-Host "[3/6] Using audio-only minimal ffmpeg.exe: $SelectedFfmpeg" -ForegroundColor Green
    } elseif (Test-Path $CachedFfmpeg) {
        $SelectedFfmpeg = $CachedFfmpeg
    } else {
        Write-Host "[3/6] Downloading static Windows x64 ffmpeg.exe (tip: run build_minimal_ffmpeg.sh in WSL/Linux for a ~4 MB audio-only build)..."
        $ZipPath = Join-Path $CacheDir "ffmpeg.zip"
        $ExtractDir = Join-Path $CacheDir "ffmpeg_extract"
        if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $ZipPath -UseBasicParsing
        Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
        $FoundFfmpeg = Get-ChildItem -Path $ExtractDir -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if (-not $FoundFfmpeg) {
            throw "Failed to locate ffmpeg.exe inside downloaded archive: $FfmpegUrl"
        }
        Copy-Item $FoundFfmpeg.FullName $CachedFfmpeg -Force
        Remove-Item $ZipPath, $ExtractDir -Recurse -Force
        $SelectedFfmpeg = $CachedFfmpeg
    }

    if ($SelectedFfmpeg) {
        $FfmpegMB = [math]::Round((Get-Item $SelectedFfmpeg).Length / 1MB, 1)
        Write-Host "      Using ffmpeg.exe: $SelectedFfmpeg ($FfmpegMB MB)"
    }

    # 4. Run PyInstaller (--onedir)
    Write-Host "[4/6] Running PyInstaller..."
    $DistAppDir = Join-Path $ProjectRoot "dist\autosub"
    if (Test-Path $DistAppDir) { Remove-Item $DistAppDir -Recurse -Force }
    Invoke-Checked $VenvPy @("-m", "PyInstaller", "--clean", "--noconfirm", (Join-Path $ProjectRoot "packaging\windows\autosub.spec")) "PyInstaller build"

    # Stage editable assets + bundled ffmpeg.exe directly beside autosub.exe
    if ($SelectedFfmpeg) {
        New-Item -ItemType Directory -Force -Path (Join-Path $DistAppDir "bin") | Out-Null
        Copy-Item $SelectedFfmpeg (Join-Path $DistAppDir "bin\ffmpeg.exe") -Force
    }
    Copy-Item (Join-Path $ProjectRoot "config") (Join-Path $DistAppDir "config") -Recurse -Force
    Copy-Item (Join-Path $ProjectRoot "profiles") (Join-Path $DistAppDir "profiles") -Recurse -Force
    Copy-Item (Join-Path $ProjectRoot "glossaries") (Join-Path $DistAppDir "glossaries") -Recurse -Force
    Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $DistAppDir ".env.example") -Force
    Copy-Item (Join-Path $ProjectRoot "README.md") (Join-Path $DistAppDir "README.md") -Force

    # 5. Smoke-test the frozen bundle
    Write-Host "[5/6] Smoke-testing dist\autosub\autosub.exe..."
    $BuiltExe = Join-Path $DistAppDir "autosub.exe"
    if ($NoBundleFfmpeg) {
        Invoke-Checked $BuiltExe @("doctor") "autosub.exe doctor"
    } else {
        Invoke-Checked $BuiltExe @("doctor", "--strict-bundle") "autosub.exe doctor --strict-bundle"
    }
    Invoke-Checked $BuiltExe @("profiles") "autosub.exe profiles"

    # Create portable ZIP
    $PortableZip = Join-Path $ProjectRoot "dist\AutoSub-$AppVersion-win64-portable.zip"
    if (Test-Path $PortableZip) { Remove-Item $PortableZip -Force }
    Compress-Archive -Path "$DistAppDir\*" -DestinationPath $PortableZip -CompressionLevel Optimal
    $ZipMB = [math]::Round((Get-Item $PortableZip).Length / 1MB, 1)
    Write-Host "      Created portable ZIP: $PortableZip ($ZipMB MB)" -ForegroundColor Green

    # 6. Compile Inno Setup Installer (.exe)
    if (-not $SkipInstaller) {
        $IsccCandidates = @(
            (Get-Command "ISCC.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }),
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        ) | Where-Object { $_ -and (Test-Path $_) }

        $Iscc = $IsccCandidates | Select-Object -First 1
        if (-not $Iscc) {
            Write-Warning "Inno Setup 6 (ISCC.exe) not found. Portable ZIP is ready; install Inno Setup 6 (`winget install JRSoftware.InnoSetup`) to build the .exe installer."
        } else {
            Write-Host "[6/6] Compiling Inno Setup installer via $Iscc..."
            Invoke-Checked $Iscc @("/DAppVersion=$AppVersion", (Join-Path $ProjectRoot "packaging\windows\autosub.iss")) "Inno Setup compilation"
            $SetupExe = Join-Path $ProjectRoot "dist\AutoSub-Setup-$AppVersion-win64.exe"
            $SetupMB = [math]::Round((Get-Item $SetupExe).Length / 1MB, 1)
            Write-Host "      Created Windows installer: $SetupExe ($SetupMB MB)" -ForegroundColor Green
        }
    }

    Write-Host "=== AutoSub Windows build complete! ===" -ForegroundColor Cyan
} finally {
    Pop-Location
}
