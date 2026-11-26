# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AutoSub Windows x64 standalone distribution (--onedir).

Bundles:
- Python runtime + PyYAML + onnxruntime + numpy + certifi
- Silero VAD ONNX model (`models/silero_vad.onnx`, zero PyTorch required)

Editable runtime assets (`config/`, `profiles/`, `glossaries/`, `bin/ffmpeg.exe`, `.env.example`)
are staged beside `autosub.exe` by `packaging/windows/build_windows.ps1` so Windows users can
customize profiles, glossaries, and API keys directly without rebuilding.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]


def _locate_silero_onnx() -> Path:
    env_p = os.environ.get("SILERO_VAD_ONNX", "").strip()
    if env_p and Path(env_p).is_file():
        return Path(env_p).resolve()
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            cand = Path(loc) / "data" / "silero_vad.onnx"
            if cand.is_file():
                return cand.resolve()
    raise FileNotFoundError(
        "未找到 silero_vad.onnx！请先运行 `pip install --no-deps silero-vad` 或设置环境变量 SILERO_VAD_ONNX。"
    )


silero_onnx = _locate_silero_onnx()

datas = [
    (str(silero_onnx), "models"),
]
datas += collect_data_files("certifi")
binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "yaml",
        "numpy",
        "onnxruntime",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        "certifi",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "torchvision",
        "tkinter",
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="autosub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="autosub",
)
