"""Profile and domain glossary loader + prompt template renderer (100% domain-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import PROFILES_DIR, PROJECT_ROOT, load_yaml


@dataclass
class ProfileConfig:
    name: str
    description: str
    source_language_hint: str
    target_language: str
    bilingual: bool
    bilingual_order: list[str]
    transcript: dict[str, Any] = field(default_factory=dict)
    translate: dict[str, Any] = field(default_factory=dict)
    glossaries: list[Path] = field(default_factory=list)
    default_vars: dict[str, str] = field(default_factory=dict)
    prompt_template: str = ""
    source_path: Path = Path()


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[tuple[str, str, dict[str, str]]]:
    """Return sorted list of (name, description, default_vars) for built-in profiles."""
    if not profiles_dir.is_dir():
        return []
    items: list[tuple[str, str, dict[str, str]]] = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        data = load_yaml(path)
        name = str(data.get("name") or path.stem)
        desc = str(data.get("description") or "").strip()
        tr = data.get("translate") or {}
        dvars = {str(k): str(v) for k, v in (tr.get("default_vars") or {}).items()}
        items.append((name, desc, dvars))
    return items


def resolve_profile_path(
    name_or_path: str | Path,
    *,
    profiles_dir: Path = PROFILES_DIR,
) -> Path:
    raw = Path(name_or_path)
    if raw.is_file():
        return raw.resolve()
    text = str(name_or_path).strip()
    candidates = [
        profiles_dir / text,
        profiles_dir / f"{text}.yaml",
        profiles_dir / f"{text}.yml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    available = ", ".join(name for name, _, _ in list_profiles(profiles_dir)) or "(none)"
    raise FileNotFoundError(
        f"找不到 Profile: {name_or_path}（可用内置 Profile: {available}，或传入任意外部 .yaml 文件路径）"
    )


def _resolve_glossary_path(raw: str | Path, *, profile_dir: Path, project_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (profile_dir, project_root, Path.cwd()):
        cand = (base / p).resolve()
        if cand.is_file():
            return cand
    return (project_root / p).resolve()


def _normalize_bilingual_order(raw_order: Any) -> list[str]:
    if not isinstance(raw_order, (list, tuple)):
        return ["target", "source"]
    cleaned = [str(x).strip().lower() for x in raw_order if str(x).strip().lower() in ("target", "source")]
    if not cleaned:
        return ["target", "source"]
    deduped: list[str] = []
    for item in cleaned:
        if item not in deduped:
            deduped.append(item)
    return deduped


def load_profile(
    name_or_path: str | Path = "default",
    *,
    profiles_dir: Path = PROFILES_DIR,
    project_root: Path = PROJECT_ROOT,
) -> ProfileConfig:
    path = resolve_profile_path(name_or_path, profiles_dir=profiles_dir)
    data = load_yaml(path)
    tr_cfg = dict(data.get("transcript") or {})
    tl_cfg = dict(data.get("translate") or {})

    raw_glossaries = tl_cfg.get("glossaries") or []
    if isinstance(raw_glossaries, str):
        raw_glossaries = [raw_glossaries]
    glossaries = [
        _resolve_glossary_path(g, profile_dir=path.parent, project_root=project_root)
        for g in raw_glossaries
        if str(g).strip()
    ]

    default_vars = {
        str(k): str(v)
        for k, v in (tl_cfg.get("default_vars") or {}).items()
    }
    prompt_template = str(tl_cfg.get("prompt_template") or "").strip()
    if not prompt_template:
        raise ValueError(f"Profile {path} 缺少 translate.prompt_template")

    return ProfileConfig(
        name=str(data.get("name") or path.stem),
        description=str(data.get("description") or "").strip(),
        source_language_hint=str(data.get("source_language_hint") or "Auto-detected spoken language").strip(),
        target_language=str(data.get("target_language") or "Simplified Chinese").strip(),
        bilingual=bool(data.get("bilingual", True)),
        bilingual_order=_normalize_bilingual_order(data.get("bilingual_order")),
        transcript=tr_cfg,
        translate=tl_cfg,
        glossaries=glossaries,
        default_vars=default_vars,
        prompt_template=prompt_template,
        source_path=path,
    )


def load_glossary(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load any domain glossary YAML (`section_name -> {target_term: [aliases...]}`)."""
    if not path.is_file():
        raise FileNotFoundError(f"找不到术语表文件: {path}")
    data = load_yaml(path)
    out: dict[str, dict[str, list[str]]] = {}
    for section, raw_sec in data.items():
        if not isinstance(raw_sec, dict):
            continue
        sec_map: dict[str, list[str]] = {}
        for target_term, raw_aliases in raw_sec.items():
            if isinstance(raw_aliases, (list, tuple)):
                aliases = [str(a).strip() for a in raw_aliases if str(a).strip()]
            elif raw_aliases is not None and str(raw_aliases).strip():
                aliases = [str(raw_aliases).strip()]
            else:
                aliases = []
            if aliases:
                sec_map[str(target_term)] = aliases
        if sec_map:
            out[str(section)] = sec_map
    return out


def merge_glossaries(paths: list[Path]) -> dict[str, dict[str, list[str]]]:
    """Merge multiple glossary YAML files in order, deduplicating aliases."""
    merged: dict[str, dict[str, list[str]]] = {}
    for path in paths:
        loaded = load_glossary(path)
        for section, mapping in loaded.items():
            target_sec = merged.setdefault(section, {})
            for target_term, aliases in mapping.items():
                existing = target_sec.setdefault(target_term, [])
                for alias in aliases:
                    if alias not in existing:
                        existing.append(alias)
    return merged


def _format_section_heading(section_key: str) -> str:
    pretty = section_key.replace("_", " ").strip()
    return pretty.title() if pretty.islower() else pretty


def render_glossary_section(glossary: dict[str, dict[str, list[str]]]) -> str:
    """Render merged glossary into a compact prompt section; returns empty string when empty."""
    lines: list[str] = []
    for section_key, mapping in glossary.items():
        entries = [f"{'/'.join(aliases)}={target}" for target, aliases in mapping.items() if aliases]
        if entries:
            lines.append(f"- {_format_section_heading(section_key)}: " + "; ".join(entries))
    if not lines:
        return ""
    return (
        "\n### Terminology Glossary\n"
        "Follow this domain terminology glossary strictly:\n"
        + "\n".join(lines)
    )


def parse_var_overrides(raw_vars: list[str] | None) -> dict[str, str]:
    """Parse repeatable CLI `--var key=value` arguments."""
    out: dict[str, str] = {}
    for item in raw_vars or []:
        if "=" not in item:
            raise ValueError(f"--var 参数格式必须为 key=value，收到: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--var 参数的 key 不能为空: {item!r}")
        out[key] = value.strip()
    return out


def render_prompt(
    profile: ProfileConfig,
    *,
    vars_override: dict[str, str] | None = None,
    glossary_text: str = "",
) -> str:
    """Render profile prompt_template with default_vars, runtime --var overrides, and glossary."""
    ctx = _SafeFormatDict(profile.default_vars)
    ctx["source_language_hint"] = profile.source_language_hint
    ctx["target_language"] = profile.target_language
    ctx["glossary_section"] = glossary_text
    if vars_override:
        for k, v in vars_override.items():
            ctx[str(k)] = str(v)
    return profile.prompt_template.format_map(ctx).strip() + "\n"
