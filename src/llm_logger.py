"""Optional rotating JSONL request/response logger for debugging LLM translation calls."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_logging_config


class LlmRequestLogger:
    """Thread-safe rotating JSONL logger capturing LLM prompts, parsed outputs, errors, and timings.

    API keys and HTTP auth headers are never passed into this logger.
    """

    def __init__(
        self,
        log_path: Path,
        *,
        enabled: bool = False,
        max_size_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.log_path = Path(log_path)
        self.enabled = bool(enabled)
        self.max_size_bytes = max(1024, int(max_size_bytes))
        self._lock = threading.Lock()

    def _rotate_if_needed(self) -> None:
        if not self.log_path.is_file():
            return
        if self.log_path.stat().st_size < self.max_size_bytes:
            return
        backup = self.log_path.with_name(f"{self.log_path.name}.old")
        try:
            if backup.exists():
                backup.unlink()
            self.log_path.replace(backup)
        except OSError:
            pass

    def log_call(
        self,
        *,
        phase: str,
        task_label: str,
        provider: str,
        model: str,
        service_tier: str = "",
        reasoning_effort: str = "",
        attempt: int = 1,
        status: str = "ok",
        elapsed_ms: int = 0,
        system_prompt: str = "",
        user_prompt: str = "",
        response: Any = None,
        usage: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Append one structured JSONL log record; silently ignores I/O failures."""
        if not self.enabled:
            return
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "phase": phase,
            "task_label": task_label,
            "provider": provider,
            "model": model,
            "service_tier": service_tier,
            "reasoning_effort": reasoning_effort,
            "attempt": attempt,
            "status": status,
            "elapsed_ms": int(elapsed_ms),
            "request": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        }
        if usage is not None:
            entry["usage"] = usage
        if response is not None:
            entry["response"] = response
        if error is not None:
            entry["error"] = error

        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:
            pass


def build_llm_logger(
    defaults: dict[str, Any],
    *,
    enabled_override: bool | None = None,
) -> LlmRequestLogger | None:
    """Create a configured `LlmRequestLogger` if enabled by config or `--log-llm`, else None."""
    cfg = get_logging_config(defaults)
    enabled = bool(cfg["llm_requests"]) if enabled_override is None else bool(enabled_override)
    if not enabled:
        return None
    max_bytes = int(float(cfg["max_size_mb"]) * 1024 * 1024)
    return LlmRequestLogger(cfg["path"], enabled=True, max_size_bytes=max_bytes)
