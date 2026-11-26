"""Engine configuration, binary resolution, and Gemini API key discovery."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.yaml"
PROFILES_DIR = PROJECT_ROOT / "profiles"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_engine_config(defaults_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if defaults_path.is_file():
        return load_yaml(defaults_path)
    return {}


def resolve_binary(configured: str, candidates: list[str]) -> str:
    cleaned = (configured or "").strip()
    if cleaned:
        return cleaned
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
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Resolve Gemini API key: config -> env vars -> project/.env -> cwd/.env."""
    if (configured_key or "").strip():
        return configured_key.strip()
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    for candidate in (project_root / ".env", Path.cwd() / ".env"):
        val = _read_key_from_dotenv(candidate)
        if val:
            return val
    return ""
