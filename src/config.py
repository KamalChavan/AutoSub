"""Engine configuration, frozen-bundle path resolution, binary lookup, and API key discovery."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

DEFAULT_PROVIDERS: dict[str, dict[str, Any]] = {
    "gemini": {
        "protocol": "gemini",
        "api_key": "",
        "api_key_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "proxy": "",
    },
    "elevenlabs": {
        "protocol": "elevenlabs",
        "api_key": "",
        "api_key_envs": ["ELEVENLABS_API_KEY"],
        "api_base": "https://api.elevenlabs.io/v1",
        "proxy": "",
    },
    "openai": {
        "protocol": "openai",
        "api_key": "",
        "api_key_envs": ["OPENAI_API_KEY"],
        "api_base": "https://api.openai.com/v1",
        "api_mode": "responses",
        "proxy": "",
    },
    "openai_compatible": {
        "protocol": "openai_compatible",
        "api_key": "",
        "api_key_envs": ["OPENAI_COMPAT_API_KEY"],
        "api_base": "",
        "api_mode": "chat_completions",
        "proxy": "",
    },
}


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


def get_network_config(defaults: dict[str, Any]) -> dict[str, Any]:
    """Return effective network defaults (`proxy`, `timeout_seconds`), honoring legacy `api.*`."""
    legacy_api = defaults.get("api") or {}
    net = defaults.get("network") or {}
    proxy = str(net.get("proxy") or legacy_api.get("proxy") or "").strip()
    raw_timeout = net.get("timeout_seconds")
    if raw_timeout is None:
        raw_timeout = legacy_api.get("timeout_seconds", 180)
    return {
        "proxy": proxy,
        "timeout_seconds": int(raw_timeout or 180),
    }


def get_providers_config(defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return normalized provider configs merged with built-in defaults and legacy `api.*`."""
    net = get_network_config(defaults)
    legacy_api = defaults.get("api") or {}
    raw_providers = defaults.get("providers") or {}
    norm_raw_providers: dict[str, Any] = {}
    if isinstance(raw_providers, dict):
        for raw_name, raw_val in raw_providers.items():
            key = str(raw_name).strip().lower()
            if key:
                norm_raw_providers[key] = raw_val

    merged: dict[str, dict[str, Any]] = {}
    all_names = list(DEFAULT_PROVIDERS.keys())
    for key in norm_raw_providers:
        if key not in all_names:
            all_names.append(key)

    for name in all_names:
        base_def = dict(DEFAULT_PROVIDERS.get(name, {}))
        user_def = norm_raw_providers.get(name) or {}
        if not isinstance(user_def, dict):
            user_def = {}
        entry = {**base_def, **user_def}

        # Backward compatibility: legacy top-level `api.*` populates `providers.gemini`
        if name == "gemini" and isinstance(legacy_api, dict):
            if not str(entry.get("api_key") or "").strip() and str(legacy_api.get("api_key") or "").strip():
                entry["api_key"] = str(legacy_api["api_key"]).strip()
            if not str(user_def.get("api_base") or "").strip() and str(legacy_api.get("api_base") or "").strip():
                entry["api_base"] = str(legacy_api["api_base"]).strip()
            if not str(user_def.get("proxy") or "").strip() and str(legacy_api.get("proxy") or "").strip():
                entry["proxy"] = str(legacy_api["proxy"]).strip()

        protocol = str(entry.get("protocol") or ("openai_compatible" if name not in DEFAULT_PROVIDERS else name)).strip().lower()
        entry["protocol"] = protocol
        entry["api_key"] = str(entry.get("api_key") or "").strip()
        raw_envs = entry.get("api_key_envs")
        if isinstance(raw_envs, str):
            entry["api_key_envs"] = [raw_envs.strip()] if raw_envs.strip() else []
        elif isinstance(raw_envs, (list, tuple)):
            entry["api_key_envs"] = [str(e).strip() for e in raw_envs if str(e).strip()]
        else:
            entry["api_key_envs"] = list(DEFAULT_PROVIDERS.get(protocol, {}).get("api_key_envs", []))
        entry["api_base"] = str(entry.get("api_base") or "").strip()
        entry["proxy"] = str(entry.get("proxy") or net["proxy"] or "").strip()
        entry["timeout_seconds"] = int(entry.get("timeout_seconds") or net["timeout_seconds"] or 180)
        if not str(entry.get("api_mode") or "").strip():
            entry["api_mode"] = "responses" if protocol == "openai" else (
                "chat_completions" if protocol == "openai_compatible" else ""
            )
        else:
            entry["api_mode"] = str(entry["api_mode"]).strip().lower()
        extra_headers = entry.get("extra_headers")
        entry["extra_headers"] = (
            {str(k): str(v) for k, v in extra_headers.items()}
            if isinstance(extra_headers, dict)
            else {}
        )
        merged[name] = entry

    return merged


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


def _read_key_from_dotenv(env_path: Path, env_names: Sequence[str]) -> str:
    if not env_path.is_file():
        return ""
    wanted = set(env_names)
    found: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        if k in wanted:
            val = v.strip().strip('"').strip("'")
            if val and k not in found:
                found[k] = val
    for name in env_names:
        if name in found:
            return found[name]
    return ""


def resolve_env_key(
    configured_key: str = "",
    *,
    env_names: Sequence[str],
    project_root: Path | None = None,
    user_config_dir: Path | None = None,
) -> str:
    """Resolve an API key: explicit config -> env vars -> cwd/.env -> app_root/.env -> user_config/.env."""
    if (configured_key or "").strip():
        return configured_key.strip()
    for env_name in env_names:
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
        val = _read_key_from_dotenv(candidate, env_names)
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
    return resolve_env_key(
        configured_key,
        env_names=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        project_root=project_root,
        user_config_dir=user_config_dir,
    )


def resolve_provider_api_key(
    provider_cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
    user_config_dir: Path | None = None,
) -> str:
    """Resolve API key for any configured provider using its `api_key` and `api_key_envs`."""
    env_names = provider_cfg.get("api_key_envs") or ()
    return resolve_env_key(
        str(provider_cfg.get("api_key") or ""),
        env_names=tuple(str(n) for n in env_names if str(n).strip()),
        project_root=project_root,
        user_config_dir=user_config_dir,
    )
