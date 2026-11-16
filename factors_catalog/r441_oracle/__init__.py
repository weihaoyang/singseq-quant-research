"""Frozen r441 oracle factor engine from commit 9cce65870a8c.

This package is intentionally isolated from the active runtime composition.
It is the semantic authority used to build and verify exact factor snapshots.
"""

from factors_catalog.r441_oracle.signal_engine import (
    build_main_wave_factor_snapshot,
)

__all__ = ["build_main_wave_factor_snapshot"]
