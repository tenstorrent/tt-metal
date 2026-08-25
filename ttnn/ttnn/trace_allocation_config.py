# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Process-start configuration for trace allocation tracking."""

import os
import warnings

from ttnn._ttnn.operations.trace import (
    trace_allocation_diagnostics_enabled,
    trace_allocation_tracking_enabled,
)


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


# Metal owns and parses these process-wide settings once. Query its cached RunTimeOptions snapshot rather than
# reading the environment independently, so TTNN and Metal cannot disagree.
TRACE_ALLOC_TRACKING = trace_allocation_tracking_enabled()
TRACE_ALLOC_DIAGNOSTICS = trace_allocation_diagnostics_enabled()
TRACE_ALLOC_REFERRER_DEPTH = (
    _env_nonnegative_int("TT_METAL_TRACE_ALLOC_REFERRER_DEPTH", 10) if TRACE_ALLOC_DIAGNOSTICS else 10
)
