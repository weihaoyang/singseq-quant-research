"""Process-launch helpers for the Windows workstation runtime."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def hidden_process_kwargs(*, new_process_group: bool = False) -> dict[str, Any]:
    """Return Popen/run kwargs that prevent console windows on Windows.

    The API is empty on POSIX so callers can use it with ``Popen`` and
    ``subprocess.run`` without platform branches at every call site.
    """
    if os.name != "nt":
        return {}
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if new_process_group:
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return {"creationflags": flags}


__all__ = ["hidden_process_kwargs"]
