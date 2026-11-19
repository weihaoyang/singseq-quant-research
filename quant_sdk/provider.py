"""Client provider compatible with full and standalone SDK trees.

The private/full client ships :mod:`quant_sdk.client`; the standalone public
tree ships :mod:`quant_client.quant_client`.  Both expose the same session/base
URL surface.  Callers import this module so package import, analytics and
preflight never retain a direct dependency on the full-only module.
"""
from __future__ import annotations

try:                                        # full client
    from quant_sdk import client as _client_module
    from quant_sdk.client import ProAPI, QuantClient
except ImportError:                         # standalone public tree
    _client_module = None
    from quant_client.quant_client import QuantClient
    from quant_client.quant_client import RemoteProAPI as ProAPI

_fallback_default: QuantClient | None = None


def init(token: str | None = None, url: str | None = None) -> QuantClient:
    """Create and retain the process default client."""
    if _client_module is not None:
        return _client_module.init(token=token, url=url)
    global _fallback_default
    _fallback_default = QuantClient(base_url=url, token=token)
    return _fallback_default


def pro_api(token: str | None = None, url: str | None = None) -> ProAPI:
    """Return a Tushare-pro-compatible data facade."""
    if _client_module is not None:
        return _client_module.pro_api(token=token, url=url)
    global _fallback_default
    if _fallback_default is None or token is not None or url is not None:
        _fallback_default = QuantClient(base_url=url, token=token)
    return _fallback_default.pro_api()


def default_client() -> QuantClient:
    """Return the existing default client or create one."""
    if _client_module is not None:
        return _client_module._DEFAULT_CLIENT or init()
    return _fallback_default or init()


__all__ = ["ProAPI", "QuantClient", "default_client", "init", "pro_api"]
