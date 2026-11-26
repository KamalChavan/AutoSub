"""Pure-Python HTTPS client for Gemini API (zero external `curl` dependency across Windows/Linux/macOS)."""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class GeminiApiError(RuntimeError):
    def __init__(self, code: str, message: str, *, http_status: int | None = None):
        super().__init__(f"Gemini API 错误 ({code}): {message}")
        self.code = str(code)
        self.message = str(message)
        self.http_status = http_status

    @property
    def is_503_transient(self) -> bool:
        c = self.code.lower()
        m = self.message.lower()
        return (
            self.http_status in (500, 502, 503, 504)
            or any(k in c for k in ("503", "500", "unavailable", "internal", "overloaded", "network"))
            or any(k in m for k in ("high demand", "temporarily unavailable", "try again later", "timed out"))
        )

    @property
    def is_429_rate_limit(self) -> bool:
        c = self.code.lower()
        m = self.message.lower()
        return (
            self.http_status == 429
            or any(k in c for k in ("429", "resource_exhausted", "too_many_requests", "rate"))
            or any(k in m for k in ("rate limit", "quota", "resource exhausted"))
        )

    @property
    def is_daily_quota_exhausted(self) -> bool:
        if not self.is_429_rate_limit:
            return False
        m = self.message.lower()
        if any(k in m for k in ("per day", "daily", "rpd", "requestsperday")):
            return True
        return self.retry_after_seconds is None

    @property
    def retry_after_seconds(self) -> int | None:
        m = re.search(r"retry in\s+(\d+)\s*s", self.message.lower())
        if m:
            val = int(m.group(1))
            if 0 < val <= 65:
                return val
        return None


def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import-not-found]
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _build_opener(proxy: str = "") -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=_build_ssl_context()),
    ]
    cleaned = (proxy or "").strip()
    if cleaned:
        handlers.insert(0, urllib.request.ProxyHandler({"http": cleaned, "https": cleaned}))
    return urllib.request.build_opener(*handlers)


def _raise_for_json_error(data: dict[str, Any], *, http_status: int | None = None) -> None:
    if "error" not in data:
        return
    err = data["error"]
    if isinstance(err, dict):
        code = err.get("code", http_status or "unknown")
        status_int = (
            http_status
            if http_status is not None
            else (int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else None)
        )
        raise GeminiApiError(str(code), str(err.get("message") or ""), http_status=status_int)
    raise GeminiApiError(str(http_status or "unknown"), str(err), http_status=http_status)


def _parse_http_error(exc: urllib.error.HTTPError) -> GeminiApiError:
    status = int(exc.code) if exc.code is not None else None
    raw = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        raw = str(exc.reason or "")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "error" in data:
                err = data["error"]
                if isinstance(err, dict):
                    return GeminiApiError(
                        str(err.get("code", status or "http_error")),
                        str(err.get("message") or raw[:300]),
                        http_status=status,
                    )
        except json.JSONDecodeError:
            pass
    return GeminiApiError(str(status or "http_error"), raw[:300] or str(exc.reason), http_status=status)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    proxy: str = "",
    timeout: int = 180,
) -> dict[str, Any]:
    """Send a JSON POST request and return parsed JSON or raise GeminiApiError."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    opener = _build_opener(proxy)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _parse_http_error(exc) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeminiApiError("network_error", str(exc)[:300], http_status=503) from exc

    data = json.loads(raw)
    if isinstance(data, dict):
        _raise_for_json_error(data)
        return data
    raise GeminiApiError("invalid_response", "API 未返回 JSON 对象", http_status=503)


def upload_resumable_file(
    file_path: Path,
    *,
    mime_type: str = "audio/mp3",
    api_key: str,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    proxy: str = "",
    timeout: int = 300,
) -> dict[str, Any]:
    """Upload a binary file using Gemini Files API resumable protocol (pure Python, streaming)."""
    size = file_path.stat().st_size
    upload_origin = api_base.rstrip("/").replace("/v1beta", "/upload/v1beta")
    opener = _build_opener(proxy)

    # Step 1: initiate resumable upload session
    meta = json.dumps({"file": {"display_name": file_path.name}}).encode("utf-8")
    start_req = urllib.request.Request(
        f"{upload_origin}/files",
        data=meta,
        method="POST",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
    )
    try:
        with opener.open(start_req, timeout=60) as resp:
            upload_url = (resp.headers.get("x-goog-upload-url") or "").strip()
    except urllib.error.HTTPError as exc:
        raise _parse_http_error(exc) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Files API 启动上传失败: {exc}") from exc

    if not upload_url:
        raise RuntimeError("Files API 未返回 x-goog-upload-url")

    # Step 2: stream file bytes to upload_url
    try:
        with file_path.open("rb") as fh:
            upload_req = urllib.request.Request(
                upload_url,
                data=fh,
                method="POST",
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                    "Content-Type": mime_type,
                },
            )
            with opener.open(upload_req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _parse_http_error(exc) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Files API 上传数据失败: {exc}") from exc

    data = json.loads(raw)
    if isinstance(data, dict):
        _raise_for_json_error(data)
        return data.get("file", data)
    raise RuntimeError("Files API 返回格式无效")


def delete_url(
    url: str,
    *,
    api_key: str,
    proxy: str = "",
    timeout: int = 30,
) -> None:
    """Best-effort HTTP DELETE (used to clean up temporary uploaded files)."""
    req = urllib.request.Request(
        url,
        method="DELETE",
        headers={"x-goog-api-key": api_key},
    )
    opener = _build_opener(proxy)
    try:
        with opener.open(req, timeout=timeout):
            pass
    except Exception:
        return
