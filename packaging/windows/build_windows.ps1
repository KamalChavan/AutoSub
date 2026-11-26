<#
.SYNOPSIS
  Build standalone Windows x64 AutoSub portable ZIP and Inno Setup (.exe) installer.

.DESCRIPTION
  Bundles:
    - Python runtime + PyYAML + onnxruntime + numpy + certifi (zero PyTorch)
    - Silero VAD ONNX model (`models/silero_vad.onnx`)
    - Static Windows x64 `ffmpeg.exe` (`bin/ffmpeg.exe`, auto-downloaded if not provided)
    - Editable `config/`, `profiles/`, `glossaries/`, `.env.example`, `README.md`

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1 -FfmpegPath C:\tools\ffmpeg.exe
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$FfmpegPath = "",
    [string]$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

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
        & $PythonExe -m venv $VenvDir
    }
    $VenvPy = Join-Path $VenvDir "Scripts\python.exe"

    Write-Host "[2/6] Installing runtime + VAD (ONNX) + PyInstaller dependencies..."
    & $VenvPy -m pip install --upgrade pip
    & $VenvPy -m pip install "PyYAML>=6.0" "onnxruntime>=1.18" "numpy>=1.26" "certifi>=2024.0" "pyinstaller>=6.10"
    & $VenvPy -m pip install --no-deps "silero-vad"

    # Run offline unit tests before packaging
    Write-Host "[2.5/6] Running offline unit tests..."
    & $VenvPy -m unittest discover -s tests -v

    # 3. Ensure static ffmpeg.exe is available
    $CacheDir = Join-Path $ProjectRoot "build\cache"
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $ResolvedFfmpeg = Join-Path $CacheDir "ffmpeg.exe"

    if ($FfmpegPath -and (Test-Path $FfmpegPath)) {
        Copy-Item $FfmpegPath $ResolvedFfmpeg -Force
    } elseif (-not (Test-Path $ResolvedFfmpeg)) {
        Write-Host "[3/6] Downloading static Windows x64 ffmpeg.exe..."
        $ZipPath = Join-Path $CacheDir "ffmpeg.zip"
        $ExtractDir = Join-Path $CacheDir "ffmpeg_extract"
        if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $ZipPath -UseBasicParsing
        Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
        $FoundFfmpeg = Get-ChildItem -Path $ExtractDir -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if (-not $FoundFfmpeg) {
            throw "Failed to locate ffmpeg.exe inside downloaded archive: $FfmpegUrl"
        }
        Copy-Item $FoundFfmpeg.FullName $ResolvedFfmpeg -Force
        Remove-Item $ZipPath, $ExtractDir -Recurse -Force
    }
    Write-Host "      Using ffmpeg.exe: $ResolvedFfmpeg"

    # 4. Run PyInstaller (--onedir)
    Write-Host "[4/6] Running PyInstaller..."
    $DistAppDir = Join-Path $ProjectRoot "dist\autosub"
    if (Test-Path $DistAppDir) { Remove-Item $DistAppDir -Recurse -Force }
    & $VenvPy -m PyInstaller --clean --noconfirm (Join-Path $ProjectRoot "packaging\windows\autosub.spec")

    # Stage editable assets + bundled ffmpeg.exe directly beside autosub.exe
    New-Item -ItemType Directory -Force -Path (Join-Path $DistAppDir "bin") | Out-Null
    Copy-Item $ResolvedFfmpeg (Join-Path $DistAppDir "bin\ffmpeg.exe") -Force
    Copy-Item (Join-Path $ProjectRoot "config") (Join-Path $DistAppDir "config") -Recurse -Force
    Copy-Item (Join-Path $ProjectRoot "profiles") (Join-Path $DistAppDir "profiles") -Recurse -Force
    Copy-Item (Join-Path $ProjectRoot "glossaries") (Join-Path $DistAppDir "glossaries") -Recurse -Force
    Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $DistAppDir ".env.example") -Force
    Copy-Item (Join-Path $ProjectRoot "README.md") (Join-Path $DistAppDir "README.md") -Force

    # 5. Smoke-test the frozen bundle strictly (requires bundled ffmpeg + Silero ONNX + profiles)
    Write-Host "[5/6] Smoke-testing dist\autosub\autosub.exe..."
    $BuiltExe = Join-Path $DistAppDir "autosub.exe"
    & $BuiltExe doctor --strict-bundle
    if ($LASTEXITCODE -ne 0) { throw "autosub.exe doctor --strict-bundle failed!" }
    & $BuiltExe profiles
    if ($LASTEXITCODE -ne 0) { throw "autosub.exe profiles failed!" }

    # Create portable ZIP
    $PortableZip = Join-Path $ProjectRoot "dist\AutoSub-$AppVersion-win64-portable.zip"
    if (Test-Path $PortableZip) { Remove-Item $PortableZip -Force }
    Compress-Archive -Path "$DistAppDir\*" -DestinationPath $PortableZip -CompressionLevel Optimal
    Write-Host "      Created portable ZIP: $PortableZip" -ForegroundColor Green

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
            & $Iscc "/DAppVersion=$AppVersion" (Join-Path $ProjectRoot "packaging\windows\autosub.iss")
            if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed!" }
            $SetupExe = Join-Path $ProjectRoot "dist\AutoSub-Setup-$AppVersion-win64.exe"
            Write-Host "      Created Windows installer: $SetupExe" -ForegroundColor Green
        }
    }

    Write-Host "=== AutoSub Windows build complete! ===" -ForegroundColor Cyan
} finally {
    Pop-Location
}
