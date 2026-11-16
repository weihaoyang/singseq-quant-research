"""Single source of truth for scalar and flag parsing helpers.

Booleans, floats and profile-state strings each had two or three byte-identical
private copies that could drift (the worst one read ``"false"`` as ``True``).
Each shape parses here once.

Environment gates were the shape where *per-site parameters* were the bug.
``env_flag`` accepted a caller-supplied token set, ``default`` and ``strip``,
so every call site carried its own vocabulary and one environment value could
open some gates and not others: ``{"1"}`` raw for migrations, ``{"1", "true",
"yes"}`` raw for startup gates, ``{"1", "true"}`` for bar gates, ``{"1",
"true", "yes", "on"}`` for cache gates.  There are now exactly two gate classes
and no per-site vocabulary:

``enable_flag``
    An opt-in gate, closed unless the operator wrote an affirmative.
``disable_flag``
    Its mirror for gates whose safe default is *on*: open unless the operator
    wrote an explicit negative.

Both always strip surrounding whitespace, so a value that survived a shell or
``.env`` quoting artifact (``"1 "``) is read the same way at every gate.  A
value outside the class's vocabulary -- unset, blank, misspelled -- leaves the
gate at its documented default, never at a third state.
"""

from __future__ import annotations

import os

__all__ = [
    "DISABLE_TOKENS",
    "ENABLE_TOKENS",
    "as_bool",
    "as_float",
    "disable_flag",
    "enable_flag",
    "normalized_state",
]

#: The affirmative spellings every opt-in gate accepts.  Anything else leaves
#: the gate closed, which is the safe direction for an opt-in.
ENABLE_TOKENS = frozenset({"1", "true", "yes", "on"})

#: The negative spellings every opt-out gate accepts.  Anything else leaves the
#: gate on, which is the documented safe default for those gates.
DISABLE_TOKENS = frozenset({"0", "false", "no", "off"})


def as_bool(value: object, *, default: bool = False) -> bool:
    """Parse a boolean-ish value strictly.

    ``bool`` passes through, ``None`` and unrecognized text fall back to
    ``default``, numbers compare against zero, and text matches the shared
    affirmative/negative vocabularies.  Unknown text never becomes truthy by
    accident.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ENABLE_TOKENS:
        return True
    if text in DISABLE_TOKENS or text == "":
        return False
    return default


def _gate_value(name: str, default: str) -> str:
    return str(os.environ.get(name, default) or "").strip().lower()


def enable_flag(name: str, *, default: str = "0") -> bool:
    """Whether the opt-in env gate ``name`` is explicitly affirmative.

    One vocabulary (``ENABLE_TOKENS``), always stripped, closed unless the
    operator wrote an affirmative.  ``default`` is the value assumed when the
    variable is unset; it is normally the closed sentinel ``"0"``.
    """
    return _gate_value(name, default) in ENABLE_TOKENS


def disable_flag(name: str, *, default: str = "1") -> bool:
    """Whether the opt-out env gate ``name`` is explicitly negative.

    The mirror of :func:`enable_flag` for gates whose safe default is *on*
    (``QUANT_USE_POV``): the gate stays on unless the operator wrote an
    explicit negative, and an unset variable keeps the documented default.
    """
    return _gate_value(name, default) in DISABLE_TOKENS


def as_float(value: object, default: float) -> float:
    """Coerce ``value`` to float, falling back to ``default`` on bad input.

    The default accepts positionally or by keyword so both historic call
    shapes (main-wave ``default=`` and replay positional) compile unchanged.
    Non-finite text still parses by site contract; callers validate ranges.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalized_state(value: object | None) -> str:
    """Normalise a profile state to its canonical lowercase form.

    Missing or blank state falls back to ``"running"``, exactly as both
    original main-wave copies did.
    """
    normalized = str(value or "").strip().lower()
    return normalized or "running"
