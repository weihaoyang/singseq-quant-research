"""Offline verification of machine-bound local-engine licenses.

The contestant's machine receives a license token and only the **public** key,
so it can verify the license without any network call.  Verification is
implemented with a dependency-free RFC 8032 Ed25519 verifier: the research
toolkit ships no crypto library, and adding one would be a large, auditable
supply-chain surface for a small check.

License token format (mirrors :mod:`server.security.engine_license`)::

    v1.<base64url(payload_json)>.<base64url(ed25519_signature)>

``payload_json`` is canonical JSON with keys
``{v, sub, machine, scope, iat, exp, jti}``.  The machine fingerprint is a
stable hash of local hardware identity, so a license issued for one machine
fails on another.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "LicenseError",
    "engine_license_required",
    "ensure_engine_license",
    "license_status",
    "load_public_key_pem",
    "machine_fingerprint",
    "verify_license",
]

_BUNDLED_PUBLIC_KEY = Path(__file__).with_name("engine_license_public.pem")


class LicenseError(Exception):
    """Raised when a license is missing, malformed, expired or out of scope."""

    def __init__(self, message: str, code: str = "LICENSE_INVALID") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Minimal, dependency-free Ed25519 verification (RFC 8032)
# ---------------------------------------------------------------------------

_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


_D = (-121665 * _inv(121666)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_D * y * y + 1)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = (x * _I) % _P
    if x % 2 != 0:
        x = _P - x
    return x


_By = (4 * _inv(5)) % _P
_Bx = _xrecover(_By)
_B = (_Bx, _By)


def _edwards_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _D * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _D * x1 * x2 * y1 * y2)
    return (x3 % _P, y3 % _P)


def _scalarmult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result: tuple[int, int] = (0, 1)
    addend = point
    while scalar > 0:
        if scalar & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        scalar >>= 1
    return result


def _on_curve(point: tuple[int, int]) -> bool:
    x, y = point
    return (-x * x + y * y - 1 - _D * x * x * y * y) % _P == 0


def _decode_point(raw: bytes) -> tuple[int, int]:
    if len(raw) != 32:
        raise ValueError("point must be 32 bytes")
    y = int.from_bytes(raw, "little") & ((1 << 255) - 1)
    x = _xrecover(y)
    if (x & 1) != ((raw[31] >> 7) & 1):
        x = _P - x
    point = (x, y)
    if not _on_curve(point):
        raise ValueError("point is not on the curve")
    return point


def _ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    try:
        a_point = _decode_point(public_key)
        r_point = _decode_point(signature[:32])
    except ValueError:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
    ) % _Q
    left = _scalarmult(_B, s)
    right = _edwards_add(r_point, _scalarmult(a_point, h))
    return left == right


# ---------------------------------------------------------------------------
# Token parsing / key loading
# ---------------------------------------------------------------------------

def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _raw_public_key(public_key_pem: str) -> bytes:
    """Extract the 32-byte raw key from an Ed25519 SubjectPublicKeyInfo PEM."""
    text = str(public_key_pem or "").strip()
    body = "".join(
        line for line in text.splitlines() if not line.startswith("-----"))
    try:
        der = base64.b64decode(body)
    except Exception as exc:
        raise LicenseError("public key is not valid PEM", code="LICENSE_KEY_INVALID") from exc
    raw = der[-32:]
    if len(raw) != 32:
        raise LicenseError("public key is not an Ed25519 key", code="LICENSE_KEY_INVALID")
    return raw


def machine_fingerprint() -> str:
    """Stable machine identity used to bind a license to one computer."""
    parts = [
        platform.node() or "",
        platform.machine() or "",
        platform.system() or "",
        f"{__import__('uuid').getnode():012x}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def load_public_key_pem(path: str | Path | None = None) -> str:
    """Resolve the engine-license public key (env -> explicit path -> bundled)."""
    inline = os.environ.get("QUANT_ENGINE_LICENSE_PUBLIC_KEY", "").strip()
    if inline:
        return inline
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_file = os.environ.get("QUANT_ENGINE_LICENSE_PUBLIC_KEY_FILE", "").strip()
    if env_file:
        candidates.append(Path(env_file).expanduser())
    candidates.append(_BUNDLED_PUBLIC_KEY)
    candidates.append(Path.home() / ".quant" / "engine_license_public.pem")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        except OSError:
            continue
    raise LicenseError(
        "未找到引擎授权公钥；请设置 QUANT_ENGINE_LICENSE_PUBLIC_KEY 或放置 engine_license_public.pem",
        code="LICENSE_KEY_MISSING")


def verify_license(token: str, *, public_key_pem: str | None = None,
                   machine: str | None = None, now: int | None = None,
                   required_scope: Iterable[str] | None = None) -> dict[str, Any]:
    """Verify a license offline; return its claims or raise :class:`LicenseError`."""
    text = str(token or "").strip()
    parts = text.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise LicenseError("授权串格式无效", code="LICENSE_FORMAT_INVALID")
    try:
        payload_bytes = _b64d(parts[1])
        signature = _b64d(parts[2])
    except Exception as exc:
        raise LicenseError("授权串编码无效", code="LICENSE_FORMAT_INVALID") from exc
    pem = public_key_pem or load_public_key_pem()
    if not _ed25519_verify(_raw_public_key(pem), payload_bytes, signature):
        raise LicenseError("授权签名校验失败", code="LICENSE_SIGNATURE_INVALID")
    try:
        claims = json.loads(payload_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise LicenseError("授权载荷无法解析", code="LICENSE_FORMAT_INVALID") from exc
    if not isinstance(claims, dict) or int(claims.get("v") or 0) != 1:
        raise LicenseError("授权版本不支持", code="LICENSE_VERSION_UNSUPPORTED")
    exp = int(claims.get("exp") or 0)
    current = int(now if now is not None else time.time())
    if exp <= current:
        raise LicenseError("授权已过期", code="LICENSE_EXPIRED")
    expected_machine = machine if machine is not None else machine_fingerprint()
    if str(claims.get("machine") or "") != expected_machine:
        raise LicenseError("授权与本机指纹不匹配", code="LICENSE_MACHINE_MISMATCH")
    if required_scope:
        have = set(str(s) for s in (claims.get("scope") or []))
        need = set(str(s) for s in required_scope)
        if not need.issubset(have):
            raise LicenseError("授权不包含所需权限", code="LICENSE_SCOPE_INSUFFICIENT")
    return claims


def license_status(token: str | None = None, *, public_key_pem: str | None = None,
                   machine: str | None = None, required_scope: Iterable[str] | None = None,
                   ) -> dict[str, Any]:
    """Non-throwing verification helper for CLI/status output."""
    token = token or os.environ.get("QUANT_ENGINE_LICENSE", "").strip()
    if not token:
        return {"valid": False, "code": "LICENSE_MISSING", "reason": "未提供授权串"}
    try:
        claims = verify_license(token, public_key_pem=public_key_pem, machine=machine,
                                required_scope=required_scope)
    except LicenseError as exc:
        return {"valid": False, "code": exc.code, "reason": str(exc)}
    return {"valid": True, "code": "OK", "claims": claims,
            "machine": machine or machine_fingerprint()}


def engine_license_required() -> bool:
    """Enforcement is opt-in via ``QUANT_ENGINE_LICENSE_REQUIRED=1``.

    The open toolkit ships engine source, so enforcement is off by default; a
    compiled/closed build sets the flag at packaging time.
    """
    return os.environ.get("QUANT_ENGINE_LICENSE_REQUIRED", "").strip().lower() in {
        "1", "true", "yes", "on"}


def ensure_engine_license(*, scope: Iterable[str] = ("local_engine",),
                          token: str | None = None) -> dict[str, Any] | None:
    """Enforce the local-engine license when the build requires it.

    Returns ``None`` when enforcement is disabled (open toolkit), otherwise the
    verified claims or raises :class:`LicenseError`.
    """
    if not engine_license_required():
        return None
    resolved = token or os.environ.get("QUANT_ENGINE_LICENSE", "").strip()
    if not resolved:
        raise LicenseError("该引擎构建需要本机授权，请设置 QUANT_ENGINE_LICENSE",
                           code="LICENSE_REQUIRED")
    return verify_license(resolved, required_scope=scope)


def _main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Offline local-engine license check")
    parser.add_argument("--license", default=None, help="授权串（或环境变量 QUANT_ENGINE_LICENSE）")
    parser.add_argument("--machine", default=None, help="覆盖本机指纹（默认自动探测）")
    parser.add_argument("--scope", default="local_engine", help="要求的权限，逗号分隔")
    parser.add_argument("--print-machine", action="store_true",
                        help="打印本机指纹（申请本机引擎授权时粘贴到门户）")
    args = parser.parse_args(argv)
    if args.print_machine:
        print(json.dumps({"machine": machine_fingerprint()}, ensure_ascii=False, indent=2))
        return 0
    scope = [s.strip() for s in str(args.scope).split(",") if s.strip()]
    status = license_status(args.license, machine=args.machine, required_scope=scope)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
