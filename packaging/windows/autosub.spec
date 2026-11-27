# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AutoSub Windows x64 slim standalone distribution (--onedir).

Zero-NumPy / Zero-PyTorch / Zero-duplicate-ORT design:
- Bundles ONLY the single standalone `onnxruntime.dll` C library (called directly via ctypes)
- Excludes `onnxruntime` Python package (`onnxruntime_pybind11_state.pyd`, -22.4 MB)
- Excludes `numpy` + `OpenBLAS` (-26.5 MB) and `flatbuffers` / `protobuf` / `sympy`
- Bundles `models/silero_vad.onnx` (2.1 MB) and `certifi` CA bundle

Editable runtime assets (`config/`, `profiles/`, `glossaries/`, `bin/ffmpeg.exe`, `.env.example`)
are staged beside `autosub.exe` by `packaging/windows/build_windows.ps1`.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

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


def _locate_onnxruntime_dll() -> Path:
    env_p = os.environ.get("ONNXRUNTIME_LIB", "").strip()
    if env_p and Path(env_p).is_file():
        return Path(env_p).resolve()
    spec = importlib.util.find_spec("onnxruntime")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            cand = Path(loc) / "capi" / "onnxruntime.dll"
            if cand.is_file():
                return cand.resolve()
    raise FileNotFoundError(
        "未找到 onnxruntime.dll！请先安装 `onnxruntime` 或设置环境变量 ONNXRUNTIME_LIB。"
    )


silero_onnx = _locate_silero_onnx()
ort_dll = _locate_onnxruntime_dll()

datas = [
    (str(silero_onnx), "models"),
]
datas += collect_data_files("certifi")

binaries = [
    (str(ort_dll), "lib"),
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "yaml",
        "certifi",
        "sqlite3",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "numpy",
        "onnxruntime",
        "silero_vad",
        "flatbuffers",
        "protobuf",
        "google.protobuf",
        "sympy",
        "mpmath",
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
