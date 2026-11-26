"""Local speech-activity detection (zero-NumPy ctypes ONNX Runtime Silero VAD + ffmpeg silencedetect fallback)."""
from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from src.config import get_app_root, get_bundle_root

SR = 16000
SILERO_WINDOW = 512
SILERO_CONTEXT = 64
SILERO_INPUT_LEN = SILERO_CONTEXT + SILERO_WINDOW  # 576
SILERO_STATE_LEN = 2 * 1 * 128  # 256

# ABI-frozen ONNX Runtime C API constants & OrtApi v1.x function table indices
_ORT_API_VERSION = 17
_ORT_LOGGING_LEVEL_ERROR = 3
_ORT_ARENA_ALLOCATOR = 0
_ORT_MEM_TYPE_DEFAULT = 0
_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7

# Exact 0-based indices verified against Microsoft onnxruntime_c_api.h (ABI-frozen since ORT 1.0)
_IDX_GET_ERROR_MESSAGE = 2
_IDX_CREATE_ENV = 3
_IDX_CREATE_SESSION = 7
_IDX_RUN = 9
_IDX_CREATE_SESSION_OPTIONS = 10
_IDX_SET_INTRA_OP_NUM_THREADS = 24
_IDX_SET_INTER_OP_NUM_THREADS = 25
_IDX_CREATE_TENSOR_WITH_DATA = 49
_IDX_GET_TENSOR_MUTABLE_DATA = 51
_IDX_CREATE_CPU_MEMORY_INFO = 69
_IDX_RELEASE_ENV = 92
_IDX_RELEASE_STATUS = 93
_IDX_RELEASE_MEMORY_INFO = 94
_IDX_RELEASE_SESSION = 95
_IDX_RELEASE_VALUE = 96
_IDX_RELEASE_SESSION_OPTIONS = 100


def find_silero_onnx_model(
    explicit_path: Path | None = None,
    *,
    app_root: Path | None = None,
    bundle_root: Path | None = None,
) -> Path | None:
    if explicit_path and explicit_path.is_file():
        return explicit_path
    env_p = os.environ.get("SILERO_VAD_ONNX", "").strip()
    if env_p and Path(env_p).is_file():
        return Path(env_p)
    for root in (app_root or get_app_root(), bundle_root or get_bundle_root()):
        for rel in (
            Path("models") / "silero_vad.onnx",
            Path("silero_vad") / "data" / "silero_vad.onnx",
        ):
            cand = root / rel
            if cand.is_file():
                return cand
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            cand = Path(loc) / "data" / "silero_vad.onnx"
            if cand.is_file():
                return cand
    return None


def _ort_lib_filenames() -> tuple[str, ...]:
    if os.name == "nt":
        return ("onnxruntime.dll",)
    if sys.platform == "darwin":
        return ("libonnxruntime.dylib",)
    return ("libonnxruntime.so", "libonnxruntime.so.1")


def find_onnxruntime_library(
    explicit_path: Path | None = None,
    *,
    app_root: Path | None = None,
    bundle_root: Path | None = None,
) -> Path | None:
    """Locate standalone ONNX Runtime C shared library (`onnxruntime.dll` / `libonnxruntime.so`)."""
    if explicit_path and explicit_path.is_file():
        return explicit_path
    env_p = os.environ.get("ONNXRUNTIME_LIB", "").strip()
    if env_p and Path(env_p).is_file():
        return Path(env_p)

    names = _ort_lib_filenames()
    roots: list[Path] = []
    for root in (app_root or get_app_root(), bundle_root or get_bundle_root()):
        if root not in roots:
            roots.append(root)

    for root in roots:
        for sub in (Path("lib"), Path("bin"), Path("onnxruntime") / "capi", Path(".")):
            base = root / sub
            if not base.is_dir():
                continue
            for name in names:
                cand = base / name
                if cand.is_file():
                    return cand
            if os.name != "nt":
                for cand in sorted(base.glob("libonnxruntime.so.*")):
                    if cand.is_file():
                        return cand

    spec = importlib.util.find_spec("onnxruntime")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            capi = Path(loc) / "capi"
            if not capi.is_dir():
                continue
            for name in names:
                cand = capi / name
                if cand.is_file():
                    return cand
            if os.name != "nt":
                for cand in sorted(capi.glob("libonnxruntime.so.*")):
                    if cand.is_file():
                        return cand
    return None


def probe_silero_vad_runtime(
    *,
    app_root: Path | None = None,
    bundle_root: Path | None = None,
) -> tuple[bool, str, Path | None, Path | None]:
    """Return `(available, backend_name, onnx_model_path, ort_lib_path)`."""
    onnx_path = find_silero_onnx_model(app_root=app_root, bundle_root=bundle_root)
    if onnx_path is None:
        return False, "missing_model", None, None

    ort_lib = find_onnxruntime_library(app_root=app_root, bundle_root=bundle_root)
    if ort_lib is not None:
        return True, "ctypes_capi", onnx_path, ort_lib

    has_py_ort = importlib.util.find_spec("onnxruntime") is not None
    has_np = importlib.util.find_spec("numpy") is not None
    if has_py_ort and has_np:
        return True, "python_onnxruntime", onnx_path, None

    return False, "missing_runtime", onnx_path, None


class _OrtApiBase(ctypes.Structure):
    _fields_ = [
        ("GetApi", ctypes.c_void_p),
        ("GetVersionString", ctypes.c_void_p),
    ]


class _SileroOrtCapiSession:
    """Zero-NumPy ctypes wrapper around ONNX Runtime C API for streaming Silero VAD inference."""

    def __init__(self, ort_lib_path: Path, onnx_model_path: Path) -> None:
        self._fn_type = ctypes.WINFUNCTYPE if os.name == "nt" else ctypes.CFUNCTYPE
        if os.name == "nt":
            self._dll = ctypes.WinDLL(str(ort_lib_path))
        else:
            self._dll = ctypes.CDLL(str(ort_lib_path))

        get_api_base = self._dll.OrtGetApiBase
        get_api_base.restype = ctypes.POINTER(_OrtApiBase)
        get_api_base.argtypes = []
        api_base_ptr = get_api_base()
        if not api_base_ptr:
            raise RuntimeError("OrtGetApiBase() returned NULL")

        get_api_fn = self._fn_type(ctypes.c_void_p, ctypes.c_uint32)(api_base_ptr.contents.GetApi)
        api_ptr = get_api_fn(_ORT_API_VERSION)
        if not api_ptr:
            api_ptr = get_api_fn(1)
        if not api_ptr:
            raise RuntimeError("OrtApiBase::GetApi() returned NULL")

        self._vtbl = ctypes.cast(api_ptr, ctypes.POINTER(ctypes.c_void_p))
        self._bind_functions()

        self._env = ctypes.c_void_p()
        self._opts = ctypes.c_void_p()
        self._session = ctypes.c_void_p()
        self._mem_info = ctypes.c_void_p()
        self._inp_val = ctypes.c_void_p()
        self._state_val = ctypes.c_void_p()
        self._sr_val = ctypes.c_void_p()

        try:
            self._check(self._CreateEnv(_ORT_LOGGING_LEVEL_ERROR, b"autosub_vad", ctypes.byref(self._env)))
            self._check(self._CreateSessionOptions(ctypes.byref(self._opts)))
            self._check(self._SetIntraOpNumThreads(self._opts, 1))
            self._check(self._SetInterOpNumThreads(self._opts, 1))
            model_arg = str(onnx_model_path) if os.name == "nt" else str(onnx_model_path).encode("utf-8")
            self._check(self._CreateSession(self._env, model_arg, self._opts, ctypes.byref(self._session)))
            self._check(self._CreateCpuMemoryInfo(_ORT_ARENA_ALLOCATOR, _ORT_MEM_TYPE_DEFAULT, ctypes.byref(self._mem_info)))

            # Pre-allocate reusable raw C buffers wrapped in-place by OrtValue tensors (zero per-frame alloc)
            self._inp_buf = (ctypes.c_float * SILERO_INPUT_LEN)()
            self._state_buf = (ctypes.c_float * SILERO_STATE_LEN)()
            self._sr_buf = (ctypes.c_int64 * 1)(SR)

            self._inp_shape = (ctypes.c_int64 * 2)(1, SILERO_INPUT_LEN)
            self._state_shape = (ctypes.c_int64 * 3)(2, 1, 128)
            self._sr_shape = (ctypes.c_int64 * 1)()

            self._check(self._CreateTensorWithData(
                self._mem_info, ctypes.byref(self._inp_buf), ctypes.sizeof(self._inp_buf),
                self._inp_shape, 2, _ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ctypes.byref(self._inp_val),
            ))
            self._check(self._CreateTensorWithData(
                self._mem_info, ctypes.byref(self._state_buf), ctypes.sizeof(self._state_buf),
                self._state_shape, 3, _ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ctypes.byref(self._state_val),
            ))
            self._check(self._CreateTensorWithData(
                self._mem_info, ctypes.byref(self._sr_buf), ctypes.sizeof(self._sr_buf),
                self._sr_shape, 0, _ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ctypes.byref(self._sr_val),
            ))

            self._in_names = (ctypes.c_char_p * 3)(b"input", b"state", b"sr")
            self._out_names = (ctypes.c_char_p * 2)(b"output", b"stateN")
            self._in_vals = (ctypes.c_void_p * 3)(self._inp_val, self._state_val, self._sr_val)
            self._out_vals = (ctypes.c_void_p * 2)(None, None)
            self._out_ptr_0 = ctypes.c_void_p()
            self._out_ptr_1 = ctypes.c_void_p()
            self._inp_addr = ctypes.addressof(self._inp_buf)
            self._inp_chunk_addr = self._inp_addr + SILERO_CONTEXT * 4
            self._inp_tail_addr = self._inp_addr + SILERO_WINDOW * 4
            self._state_addr = ctypes.addressof(self._state_buf)
            self._context_bytes = SILERO_CONTEXT * 4
            self._window_bytes = SILERO_WINDOW * 4
            self._state_bytes = SILERO_STATE_LEN * 4
        except BaseException:
            self.close()
            raise

    def _fn(self, idx: int, restype, *argtypes):
        return self._fn_type(restype, *argtypes)(self._vtbl[idx])

    def _bind_functions(self) -> None:
        vp = ctypes.c_void_p
        pvp = ctypes.POINTER(vp)
        pi64 = ctypes.POINTER(ctypes.c_int64)
        pchar = ctypes.POINTER(ctypes.c_char_p)
        model_path_t = ctypes.c_wchar_p if os.name == "nt" else ctypes.c_char_p

        self._GetErrorMessage = self._fn(_IDX_GET_ERROR_MESSAGE, ctypes.c_char_p, vp)
        self._CreateEnv = self._fn(_IDX_CREATE_ENV, vp, ctypes.c_int, ctypes.c_char_p, pvp)
        self._CreateSessionOptions = self._fn(_IDX_CREATE_SESSION_OPTIONS, vp, pvp)
        self._SetIntraOpNumThreads = self._fn(_IDX_SET_INTRA_OP_NUM_THREADS, vp, vp, ctypes.c_int)
        self._SetInterOpNumThreads = self._fn(_IDX_SET_INTER_OP_NUM_THREADS, vp, vp, ctypes.c_int)
        self._CreateSession = self._fn(_IDX_CREATE_SESSION, vp, vp, model_path_t, vp, pvp)
        self._CreateCpuMemoryInfo = self._fn(_IDX_CREATE_CPU_MEMORY_INFO, vp, ctypes.c_int, ctypes.c_int, pvp)
        self._CreateTensorWithData = self._fn(
            _IDX_CREATE_TENSOR_WITH_DATA, vp, vp, vp, ctypes.c_size_t, pi64, ctypes.c_size_t, ctypes.c_int, pvp,
        )
        self._Run = self._fn(
            _IDX_RUN, vp, vp, vp, pchar, pvp, ctypes.c_size_t, pchar, ctypes.c_size_t, pvp,
        )
        self._GetTensorMutableData = self._fn(_IDX_GET_TENSOR_MUTABLE_DATA, vp, vp, pvp)
        self._ReleaseEnv = self._fn(_IDX_RELEASE_ENV, None, vp)
        self._ReleaseStatus = self._fn(_IDX_RELEASE_STATUS, None, vp)
        self._ReleaseMemoryInfo = self._fn(_IDX_RELEASE_MEMORY_INFO, None, vp)
        self._ReleaseSession = self._fn(_IDX_RELEASE_SESSION, None, vp)
        self._ReleaseValue = self._fn(_IDX_RELEASE_VALUE, None, vp)
        self._ReleaseSessionOptions = self._fn(_IDX_RELEASE_SESSION_OPTIONS, None, vp)

    def _check(self, status: int | None) -> None:
        if status:
            try:
                raw_msg = self._GetErrorMessage(status)
                msg = raw_msg.decode("utf-8", errors="replace") if raw_msg else "unknown ORT error"
            finally:
                self._ReleaseStatus(status)
            raise RuntimeError(f"ONNX Runtime C API error: {msg}")

    def step_from_addr(self, chunk_addr: int) -> float:
        """Run one 512-sample float32 audio window at raw memory address `chunk_addr`."""
        ctypes.memmove(self._inp_chunk_addr, chunk_addr, self._window_bytes)
        self._out_vals[0] = None
        self._out_vals[1] = None
        status = self._Run(
            self._session, None, self._in_names, self._in_vals, 3, self._out_names, 2, self._out_vals,
        )
        if status:
            self._check(status)

        o0 = self._out_vals[0]
        o1 = self._out_vals[1]
        try:
            self._check(self._GetTensorMutableData(o0, ctypes.byref(self._out_ptr_0)))
            prob = ctypes.c_float.from_address(self._out_ptr_0.value).value
            self._check(self._GetTensorMutableData(o1, ctypes.byref(self._out_ptr_1)))
            ctypes.memmove(self._state_addr, self._out_ptr_1.value, self._state_bytes)
        finally:
            if o0:
                self._ReleaseValue(o0)
            if o1:
                self._ReleaseValue(o1)

        # Slide last 64 samples into the front context window for the next frame
        ctypes.memmove(self._inp_addr, self._inp_tail_addr, self._context_bytes)
        return prob

    def close(self) -> None:
        for attr, releaser_name in (
            ("_inp_val", "_ReleaseValue"),
            ("_state_val", "_ReleaseValue"),
            ("_sr_val", "_ReleaseValue"),
            ("_mem_info", "_ReleaseMemoryInfo"),
            ("_session", "_ReleaseSession"),
            ("_opts", "_ReleaseSessionOptions"),
            ("_env", "_ReleaseEnv"),
        ):
            val = getattr(self, attr, None)
            if val and val.value:
                releaser = getattr(self, releaser_name, None)
                if releaser is not None:
                    releaser(val)
                val.value = None

    def __enter__(self) -> _SileroOrtCapiSession:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _finalize_vad_segments(
    raw_segments: list[list[int]],
    *,
    total_samples: int,
    speech_pad_ms: int,
    merge_gap_ms: int,
) -> tuple[list[tuple[float, float]], float]:
    pad_samples = int(SR * speech_pad_ms / 1000)
    merge_gap_samples = int(SR * merge_gap_ms / 1000)
    merged: list[list[int]] = []
    for s, e in raw_segments:
        ps = max(0, s - pad_samples)
        pe = min(total_samples, e + pad_samples)
        if merged and ps <= merged[-1][1] + merge_gap_samples:
            merged[-1][1] = max(merged[-1][1], pe)
        else:
            merged.append([ps, pe])
    return [(round(s / SR, 3), round(e / SR, 3)) for s, e in merged], total_samples / SR


def _detect_silero_via_capi(
    pcm_path: Path,
    *,
    onnx_path: Path,
    ort_lib_path: Path,
    threshold: float,
    neg_threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    merge_gap_ms: int,
) -> tuple[list[tuple[float, float]], float]:
    total_samples = pcm_path.stat().st_size // 4
    min_speech_samples = int(SR * min_speech_ms / 1000)
    min_silence_samples = int(SR * min_silence_ms / 1000)

    triggered = False
    temp_end = 0
    cur_start = 0
    raw_segments: list[list[int]] = []
    block_bytes = SILERO_WINDOW * 1024 * 4  # ~32.7s per streaming block (~2 MB RAM)
    window_bytes = SILERO_WINDOW * 4
    sample_cursor = 0

    with _SileroOrtCapiSession(ort_lib_path, onnx_path) as sess:
        buf = (ctypes.c_char * block_bytes)()
        buf_base = ctypes.addressof(buf)
        step_fn = sess.step_from_addr

        with pcm_path.open("rb") as fh:
            while True:
                n_bytes = fh.readinto(buf)
                if not n_bytes:
                    break
                n_win = n_bytes // window_bytes
                for i in range(n_win):
                    prob = step_fn(buf_base + i * window_bytes)
                    pos = sample_cursor + i * SILERO_WINDOW

                    if prob >= threshold and temp_end:
                        temp_end = 0
                    if prob >= threshold and not triggered:
                        triggered = True
                        cur_start = pos
                        continue
                    if prob < neg_threshold and triggered:
                        if not temp_end:
                            temp_end = pos
                        if pos - temp_end >= min_silence_samples:
                            if temp_end - cur_start >= min_speech_samples:
                                raw_segments.append([cur_start, temp_end])
                            cur_start = 0
                            temp_end = 0
                            triggered = False
                sample_cursor += n_bytes // 4

    if triggered and (total_samples - cur_start >= min_speech_samples):
        raw_segments.append([cur_start, total_samples])

    return _finalize_vad_segments(
        raw_segments,
        total_samples=total_samples,
        speech_pad_ms=speech_pad_ms,
        merge_gap_ms=merge_gap_ms,
    )


def _detect_silero_via_python_ort(
    pcm_path: Path,
    *,
    onnx_path: Path,
    threshold: float,
    neg_threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    merge_gap_ms: int,
) -> tuple[list[tuple[float, float]], float]:
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
    sr_arr = np.array(SR, dtype=np.int64)

    total_samples = pcm_path.stat().st_size // 4
    min_speech_samples = int(SR * min_speech_ms / 1000)
    min_silence_samples = int(SR * min_silence_ms / 1000)

    triggered = False
    temp_end = 0
    cur_start = 0
    raw_segments: list[list[int]] = []
    block_samples = SILERO_WINDOW * 1024
    sample_cursor = 0

    with pcm_path.open("rb") as fh:
        while True:
            raw = fh.read(block_samples * 4)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=np.float32)
            n_win = len(arr) // SILERO_WINDOW
            for i in range(n_win):
                chunk = arr[i * SILERO_WINDOW : (i + 1) * SILERO_WINDOW].reshape(1, SILERO_WINDOW)
                inp = np.concatenate([context, chunk], axis=1)
                out, state = sess.run(None, {"input": inp, "state": state, "sr": sr_arr})
                context = chunk[:, -SILERO_CONTEXT:]
                prob = float(out[0][0])
                pos = sample_cursor + i * SILERO_WINDOW

                if prob >= threshold and temp_end:
                    temp_end = 0
                if prob >= threshold and not triggered:
                    triggered = True
                    cur_start = pos
                    continue
                if prob < neg_threshold and triggered:
                    if not temp_end:
                        temp_end = pos
                    if pos - temp_end >= min_silence_samples:
                        if temp_end - cur_start >= min_speech_samples:
                            raw_segments.append([cur_start, temp_end])
                        cur_start = 0
                        temp_end = 0
                        triggered = False
            sample_cursor += len(arr)

    if triggered and (total_samples - cur_start >= min_speech_samples):
        raw_segments.append([cur_start, total_samples])

    return _finalize_vad_segments(
        raw_segments,
        total_samples=total_samples,
        speech_pad_ms=speech_pad_ms,
        merge_gap_ms=merge_gap_ms,
    )


def detect_speech_segments_silero(
    pcm_path: Path,
    *,
    onnx_path: Path,
    ort_lib_path: Path | None = None,
    threshold: float = 0.35,
    neg_threshold: float = 0.20,
    min_speech_ms: int = 200,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 500,
    merge_gap_ms: int = 600,
) -> tuple[list[tuple[float, float]], float]:
    """Streaming pure-ONNX Silero VAD (prefers zero-NumPy ctypes C API; falls back to Python onnxruntime)."""
    resolved_lib = ort_lib_path or find_onnxruntime_library()
    if resolved_lib is not None:
        return _detect_silero_via_capi(
            pcm_path,
            onnx_path=onnx_path,
            ort_lib_path=resolved_lib,
            threshold=threshold,
            neg_threshold=neg_threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            merge_gap_ms=merge_gap_ms,
        )
    return _detect_silero_via_python_ort(
        pcm_path,
        onnx_path=onnx_path,
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        merge_gap_ms=merge_gap_ms,
    )


def detect_speech_segments_silencedetect(
    pcm_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    noise_db: float = -30.0,
    min_silence_s: float = 1.5,
    speech_pad_s: float = 0.5,
) -> tuple[list[tuple[float, float]], float]:
    """Zero-dependency ffmpeg silencedetect fallback when ONNX Runtime/Silero is unavailable."""
    total_dur = (pcm_path.stat().st_size // 4) / SR
    res = subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-nostdin",
            "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(pcm_path),
            "-af", f"silencedetect=noise={noise_db:g}dB:d={min_silence_s:g}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    silences: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in res.stderr.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m1:
            cur_start = float(m1.group(1))
        if m2 and cur_start is not None:
            silences.append((cur_start, float(m2.group(1))))
            cur_start = None
    if cur_start is not None:
        silences.append((cur_start, total_dur))

    kept: list[list[float]] = []
    pos = 0.0
    for s0, s1 in silences:
        a = max(0.0, pos - speech_pad_s)
        b = min(total_dur, s0 + speech_pad_s)
        if b > a:
            if kept and a <= kept[-1][1]:
                kept[-1][1] = max(kept[-1][1], b)
            else:
                kept.append([a, b])
        pos = s1
    if pos < total_dur:
        a = max(0.0, pos - speech_pad_s)
        b = total_dur
        if kept and a <= kept[-1][1]:
            kept[-1][1] = max(kept[-1][1], b)
        else:
            kept.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in kept], total_dur


def pack_segments_into_chunks(
    segments: list[tuple[float, float]],
    *,
    max_chunk_seconds: float = 1680.0,
    silence_gap_s: float = 0.5,
) -> list[list[dict[str, float | int]]]:
    """Pack speech segments into <= max_chunk_seconds chunks with exact [compact -> orig] maps."""
    if max_chunk_seconds <= 10.0:
        raise ValueError("max_chunk_seconds 必须大于 10 秒")
    chunks: list[list[dict[str, float | int]]] = []
    cur_chunk: list[dict[str, float | int]] = []
    cursor = 0.0
    global_idx = 0

    for orig_s, orig_e in segments:
        pos = orig_s
        while pos < orig_e:
            sub_end = min(orig_e, pos + max_chunk_seconds)
            dur = sub_end - pos
            needed = dur if not cur_chunk else (silence_gap_s + dur)
            if cur_chunk and cursor + needed > max_chunk_seconds:
                chunks.append(cur_chunk)
                cur_chunk = []
                cursor = 0.0
            if cur_chunk:
                cursor += silence_gap_s
            cur_chunk.append({
                "idx": global_idx,
                "compact_start": round(cursor, 4),
                "compact_end": round(cursor + dur, 4),
                "orig_start": round(pos, 4),
                "orig_end": round(sub_end, 4),
            })
            cursor += dur
            global_idx += 1
            pos = sub_end

    if cur_chunk:
        chunks.append(cur_chunk)
    return chunks


def render_chunk_mp3(
    pcm_path: Path,
    chunk_map: list[dict[str, float | int]],
    out_mp3: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    silence_gap_s: float = 0.5,
) -> None:
    """Stream selected PCM segments into a compact MP3 with constant low memory usage."""
    gap_bytes = b"\x00" * (int(round(SR * silence_gap_s)) * 4)
    total_bytes = pcm_path.stat().st_size
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libmp3lame", "-b:a", "48k", str(out_mp3),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        with pcm_path.open("rb") as src_fh:
            for i, seg in enumerate(chunk_map):
                start_byte = max(0, min(total_bytes, int(round(float(seg["orig_start"]) * SR)) * 4))
                end_byte = max(start_byte, min(total_bytes, int(round(float(seg["orig_end"]) * SR)) * 4))
                src_fh.seek(start_byte)
                remaining = end_byte - start_byte
                while remaining > 0:
                    block = src_fh.read(min(65536, remaining))
                    if not block:
                        break
                    proc.stdin.write(block)
                    remaining -= len(block)
                if i + 1 < len(chunk_map) and gap_bytes:
                    proc.stdin.write(gap_bytes)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg 编码压缩音频失败 ({rc}): {stderr[:300]}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
