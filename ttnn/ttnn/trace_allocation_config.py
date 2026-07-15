# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Process-start configuration for trace allocation tracking."""

import os
import warnings


def _env_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        warnings.warn(f"{name} must be a non-negative integer; using {default}", stacklevel=2)
        return default


# These values are intentionally captured once, when ttnn starts importing.
TRACE_ALLOC_TRACKING = _env_enabled("TT_METAL_TRACE_ALLOC_TRACKING")
TRACE_ALLOC_DIAGNOSTICS = TRACE_ALLOC_TRACKING and _env_enabled("TT_METAL_TRACE_ALLOC_TRACEBACKS")
TRACE_ALLOC_REFERRER_DEPTH = (
    _env_nonnegative_int("TT_METAL_TRACE_ALLOC_REFERRER_DEPTH", 10) if TRACE_ALLOC_DIAGNOSTICS else 10
)
