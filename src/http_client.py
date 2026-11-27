"""Pure-Python HTTPS client for multi-provider AI APIs (zero external `curl`/SDK dependency)."""
from __future__ import annotations

import json
import re
import ssl
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence

_MULTIPART_MEMORY_LIMIT = 16 * 1024 * 1024  # 16 MB; larger uploads spill to temp file


class ProviderApiError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        provider: str = "API",
        retry_after: int | None = None,
        error_type: str = "",
    ):
        super().__init__(f"{provider} API 错误 ({code}): {message}")
        self.code = str(code)
        self.message = str(message)
        self.http_status = http_status
        self.provider = provider
        self.retry_after = retry_after
        self.error_type = str(error_type or "")

    @property
    def is_503_transient(self) -> bool:
        c = f"{self.code} {self.error_type}".lower()
        m = self.message.lower()
        return (
            self.http_status in (500, 502, 503, 504, 529)
            or any(
                k in c
                for k in (
                    "503",
                    "500",
                    "unavailable",
                    "internal",
                    "overloaded",
                    "network",
                    "server_error",
                    "system_busy",
                )
            )
            or any(
                k in m
                for k in (
                    "high demand",
                    "temporarily unavailable",
                    "try again later",
                    "timed out",
                    "system is currently busy",
                )
            )
        )

    @property
    def is_429_rate_limit(self) -> bool:
        c = f"{self.code} {self.error_type}".lower()
        m = self.message.lower()
        return (
            self.http_status == 429
            or any(
                k in c
                for k in (
                    "429",
                    "resource_exhausted",
                    "too_many_requests",
                    "rate_limit",
                    "concurrent_limit",
                    "too_many_concurrent",
                    "insufficient_quota",
                    "rate",
                )
            )
            or any(
                k in m
                for k in (
                    "rate limit",
                    "quota",
                    "resource exhausted",
                    "too many concurrent",
                    "concurrent requests",
                )
            )
        )

    @property
    def is_billing_or_quota_exhausted(self) -> bool:
        c = f"{self.code} {self.error_type}".lower()
        m = self.message.lower()
        if self.http_status == 402:
            return True
        return any(
            k in c or k in m
            for k in (
                "insufficient_quota",
                "insufficient_credits",
                "quota_exceeded",
                "billing_hard_limit",
                "does not have enough credits",
                "exceeded your current quota",
            )
        )

    @property
    def is_daily_quota_exhausted(self) -> bool:
        if self.is_billing_or_quota_exhausted:
            return True
        if not self.is_429_rate_limit:
            return False
        c = f"{self.code} {self.error_type}".lower()
        m = self.message.lower()
        if any(k in m for k in ("per day", "daily", "rpd", "requestsperday")):
            return True
        if any(k in c or k in m for k in ("concurrent", "system_busy")):
            return False
        return self.retry_after_seconds is None

    @property
    def retry_after_seconds(self) -> int | None:
        if self.retry_after is not None and 0 < self.retry_after <= 120:
            return self.retry_after
        m = re.search(r"(?:retry|try again) in\s+(\d+(?:\.\d+)?)\s*s", self.message.lower())
        if m:
            val = max(1, int(round(float(m.group(1)))))
            if 0 < val <= 65:
                return val
        return None


# Backward-compatible alias for existing callers and tests
GeminiApiError = ProviderApiError


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


def _parse_retry_after_header(headers: Any) -> int | None:
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if not raw:
            continue
        text = str(raw).strip().lower()
        if text.endswith("ms"):
            try:
                ms = float(text[:-2])
                if ms > 0:
                    return max(1, int(round(ms / 1000.0)))
            except ValueError:
                continue
        elif text.endswith("s"):
            try:
                sec = float(text[:-1])
                if sec > 0:
                    return max(1, int(round(sec)))
            except ValueError:
                continue
        else:
            try:
                sec = float(text)
                if sec > 0:
                    return max(1, int(round(sec)))
            except ValueError:
                continue
    return None


def _raise_for_json_error(
    data: dict[str, Any],
    *,
    http_status: int | None = None,
    provider_name: str = "API",
) -> None:
    if "error" not in data or data["error"] is None:
        return
    err = data["error"]
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or http_status or "unknown"
        err_type = str(err.get("type") or "")
        status_int = (
            http_status
            if http_status is not None
            else (int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else None)
        )
        raise ProviderApiError(
            str(code),
            str(err.get("message") or ""),
            http_status=status_int,
            provider=provider_name,
            error_type=err_type,
        )
    raise ProviderApiError(
        str(http_status or "unknown"),
        str(err),
        http_status=http_status,
        provider=provider_name,
    )


def _parse_http_error(
    exc: urllib.error.HTTPError,
    *,
    provider_name: str = "API",
) -> ProviderApiError:
    status = int(exc.code) if exc.code is not None else None
    retry_after = _parse_retry_after_header(getattr(exc, "headers", None))
    raw = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        raw = str(exc.reason or "")

    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # Gemini / OpenAI style: {"error": {"code": ..., "type": ..., "message": ...}}
                if "error" in data and data["error"] is not None:
                    err = data["error"]
                    if isinstance(err, dict):
                        code = err.get("code") or err.get("type") or status or "http_error"
                        err_type = str(err.get("type") or "")
                        return ProviderApiError(
                            str(code),
                            str(err.get("message") or raw[:300]),
                            http_status=status,
                            provider=provider_name,
                            retry_after=retry_after,
                            error_type=err_type,
                        )
                    return ProviderApiError(
                        str(status or "http_error"),
                        str(err)[:300],
                        http_status=status,
                        provider=provider_name,
                        retry_after=retry_after,
                    )
                # ElevenLabs / FastAPI style: {"detail": {"status": ..., "message": ...} | str | list}
                if "detail" in data and data["detail"] is not None:
                    detail = data["detail"]
                    if isinstance(detail, dict):
                        code = detail.get("status") or detail.get("code") or detail.get("type") or status or "http_error"
                        err_type = str(detail.get("type") or detail.get("status") or "")
                        msg = str(detail.get("message") or raw[:300])
                        return ProviderApiError(
                            str(code),
                            msg,
                            http_status=status,
                            provider=provider_name,
                            retry_after=retry_after,
                            error_type=err_type,
                        )
                    if isinstance(detail, list):
                        msgs = []
                        for item in detail[:3]:
                            if isinstance(item, dict):
                                loc = ".".join(str(p) for p in (item.get("loc") or []))
                                m = str(item.get("msg") or item)
                                msgs.append(f"{loc}: {m}" if loc else m)
                            else:
                                msgs.append(str(item))
                        return ProviderApiError(
                            str(status or "validation_error"),
                            "; ".join(msgs) or raw[:300],
                            http_status=status,
                            provider=provider_name,
                            retry_after=retry_after,
                            error_type="validation_error",
                        )
                    return ProviderApiError(
                        str(status or "http_error"),
                        str(detail)[:300],
                        http_status=status,
                        provider=provider_name,
                        retry_after=retry_after,
                    )
        except json.JSONDecodeError:
            pass
    return ProviderApiError(
        str(status or "http_error"),
        raw[:300] or str(exc.reason),
        http_status=status,
        provider=provider_name,
        retry_after=retry_after,
    )


def get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    proxy: str = "",
    timeout: int = 30,
    provider_name: str = "API",
) -> dict[str, Any]:
    """Send an HTTP GET request and return parsed JSON or raise ProviderApiError."""
    req = urllib.request.Request(url, method="GET", headers=dict(headers or {}))
    opener = _build_opener(proxy)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _parse_http_error(exc, provider_name=provider_name) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderApiError("network_error", str(exc)[:300], http_status=503, provider=provider_name) from exc

    data = json.loads(raw)
    if isinstance(data, dict):
        _raise_for_json_error(data, provider_name=provider_name)
        return data
    raise ProviderApiError("invalid_response", "API 未返回 JSON 对象", http_status=503, provider=provider_name)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "",
    headers: dict[str, str] | None = None,
    proxy: str = "",
    timeout: int = 180,
    provider_name: str = "Gemini",
) -> dict[str, Any]:
    """Send a JSON POST request and return parsed JSON or raise ProviderApiError."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers: dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    elif api_key:
        req_headers["x-goog-api-key"] = api_key

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=req_headers,
    )
    opener = _build_opener(proxy)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _parse_http_error(exc, provider_name=provider_name) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderApiError("network_error", str(exc)[:300], http_status=503, provider=provider_name) from exc

    data = json.loads(raw)
    if isinstance(data, dict):
        _raise_for_json_error(data, provider_name=provider_name)
        return data
    raise ProviderApiError("invalid_response", "API 未返回 JSON 对象", http_status=503, provider=provider_name)


def _escape_multipart_token(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def build_multipart_payload(
    *,
    fields: Sequence[tuple[str, str]],
    files: Sequence[tuple[str, Path, str]],
    boundary: str | None = None,
) -> tuple[str, bytes]:
    """Build an in-memory `multipart/form-data` payload (supports repeated field names)."""
    bnd = boundary or f"----AutoSubBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        safe_name = _escape_multipart_token(str(name))
        chunks.append(
            f"--{bnd}\r\n"
            f'Content-Disposition: form-data; name="{safe_name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    for field_name, file_path, mime_type in files:
        safe_name = _escape_multipart_token(str(field_name))
        safe_fname = _escape_multipart_token(file_path.name)
        chunks.append(
            f"--{bnd}\r\n"
            f'Content-Disposition: form-data; name="{safe_name}"; filename="{safe_fname}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8")
        )
        chunks.append(file_path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{bnd}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={bnd}", b"".join(chunks)


def _stream_multipart_to_file(
    fh: Any,
    *,
    fields: Sequence[tuple[str, str]],
    files: Sequence[tuple[str, Path, str]],
    boundary: str,
) -> int:
    total = 0

    def write_bytes(b: bytes) -> None:
        nonlocal total
        fh.write(b)
        total += len(b)

    for name, value in fields:
        safe_name = _escape_multipart_token(str(name))
        write_bytes(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{safe_name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    for field_name, file_path, mime_type in files:
        safe_name = _escape_multipart_token(str(field_name))
        safe_fname = _escape_multipart_token(file_path.name)
        write_bytes(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{safe_name}"; filename="{safe_fname}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8")
        )
        with file_path.open("rb") as src:
            while True:
                buf = src.read(65536)
                if not buf:
                    break
                write_bytes(buf)
        write_bytes(b"\r\n")
    write_bytes(f"--{boundary}--\r\n".encode("utf-8"))
    fh.flush()
    fh.seek(0)
    return total


def post_multipart(
    url: str,
    *,
    fields: Sequence[tuple[str, str]] = (),
    files: Sequence[tuple[str, Path, str]] = (),
    headers: dict[str, str] | None = None,
    proxy: str = "",
    timeout: int = 360,
    provider_name: str = "API",
) -> dict[str, Any]:
    """Send a `multipart/form-data` POST request (spills large files to disk to keep RAM low)."""
    total_file_bytes = sum(p.stat().st_size for _, p, _ in files)
    opener = _build_opener(proxy)
    req_headers: dict[str, str] = dict(headers or {})

    if total_file_bytes <= _MULTIPART_MEMORY_LIMIT:
        content_type, body = build_multipart_payload(fields=fields, files=files)
        req_headers["Content-Type"] = content_type
        req_headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body, method="POST", headers=req_headers)
        try:
            with opener.open(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise _parse_http_error(exc, provider_name=provider_name) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderApiError("network_error", str(exc)[:300], http_status=503, provider=provider_name) from exc
    else:
        bnd = f"----AutoSubBoundary{uuid.uuid4().hex}"
        with tempfile.TemporaryFile() as spool:
            content_len = _stream_multipart_to_file(spool, fields=fields, files=files, boundary=bnd)
            req_headers["Content-Type"] = f"multipart/form-data; boundary={bnd}"
            req_headers["Content-Length"] = str(content_len)
            req = urllib.request.Request(url, data=spool, method="POST", headers=req_headers)
            try:
                with opener.open(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                raise _parse_http_error(exc, provider_name=provider_name) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise ProviderApiError("network_error", str(exc)[:300], http_status=503, provider=provider_name) from exc

    data = json.loads(raw)
    if isinstance(data, dict):
        _raise_for_json_error(data, provider_name=provider_name)
        return data
    raise ProviderApiError("invalid_response", "API 未返回 JSON 对象", http_status=503, provider=provider_name)


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
        raise _parse_http_error(exc, provider_name="Gemini") from exc
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
        raise _parse_http_error(exc, provider_name="Gemini") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Files API 上传数据失败: {exc}") from exc

    data = json.loads(raw)
    if isinstance(data, dict):
        _raise_for_json_error(data, provider_name="Gemini")
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
