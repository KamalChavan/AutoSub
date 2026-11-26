"""Engine configuration, frozen-bundle path resolution, binary lookup, and API key discovery."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller packaged executable."""
    return bool(getattr(sys, "frozen", False))


def get_app_root() -> Path:
    """Editable application root (directory containing autosub.exe when frozen, repo root in dev)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_bundle_root() -> Path:
    """Read-only bundled internal resource root (_MEIPASS when frozen, repo root in dev)."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_user_config_dir() -> Path | None:
    """Per-user config directory (`%APPDATA%/AutoSub` on Windows, `~/.config/autosub` elsewhere)."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata) / "AutoSub"
        try:
            return Path.home() / "AppData" / "Roaming" / "AutoSub"
        except RuntimeError:
            return None
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg) / "autosub"
    try:
        return Path.home() / ".config" / "autosub"
    except RuntimeError:
        return None


PROJECT_ROOT = get_app_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.yaml"
PROFILES_DIR = PROJECT_ROOT / "profiles"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_engine_config(defaults_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if defaults_path.is_file():
        return load_yaml(defaults_path)
    bundled_fallback = get_bundle_root() / "config" / "defaults.yaml"
    if bundled_fallback.is_file():
        return load_yaml(bundled_fallback)
    return {}


def resolve_binary(
    configured: str,
    candidates: list[str],
    *,
    app_root: Path | None = None,
    bundle_root: Path | None = None,
) -> str:
    """Resolve an external binary: explicit config -> bundled `bin/` -> system PATH."""
    cleaned = (configured or "").strip()
    if cleaned:
        return cleaned

    roots: list[Path] = []
    for root in (app_root or get_app_root(), bundle_root or get_bundle_root()):
        if root not in roots:
            roots.append(root)

    for root in roots:
        for sub in ("bin", ""):
            base = (root / sub) if sub else root
            for name in candidates:
                cand = base / name
                if cand.is_file():
                    return str(cand)

    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return candidates[0]


def _read_key_from_dotenv(env_path: Path) -> str:
    if not env_path.is_file():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = v.strip().strip('"').strip("'")
            if val:
                return val
    return ""


def resolve_api_key(
    configured_key: str = "",
    *,
    project_root: Path | None = None,
    user_config_dir: Path | None = None,
) -> str:
    """Resolve Gemini API key: config -> env vars -> cwd/.env -> app_root/.env -> user_config/.env."""
    if (configured_key or "").strip():
        return configured_key.strip()
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    root = project_root or get_app_root()
    uconf = user_config_dir if user_config_dir is not None else get_user_config_dir()
    candidates = [Path.cwd() / ".env", root / ".env"]
    if uconf is not None:
        candidates.append(uconf / ".env")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        val = _read_key_from_dotenv(candidate)
        if val:
            return val
    return ""
