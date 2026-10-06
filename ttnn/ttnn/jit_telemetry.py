# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""JIT build telemetry, read in-process.

Values map token names (e.g. ``program_config_size.total.TENSIX``) to
``{"unit", "count", "total", "min", "max"}``. Token names may change; treat a missing one as unknown.

    with ttnn.jit_telemetry.capture() as cap:
        run_some_programs()
    largest = cap.stats.get("program_config_size.total.TENSIX", {}).get("max")
"""

import contextlib

import ttnn


def snapshot() -> dict:
    """Process-wide values of every token."""
    return ttnn._ttnn.jit_telemetry.snapshot()


class Capture:
    """Result of :func:`capture`; ``stats`` is filled when the block exits."""

    def __init__(self):
        self.stats: dict = {}


@contextlib.contextmanager
def capture():
    """Collect what every token records inside the block, from any thread. Captures may nest."""
    cap = Capture()
    capture_id = ttnn._ttnn.jit_telemetry.begin_capture()
    try:
        yield cap
    finally:
        cap.stats = ttnn._ttnn.jit_telemetry.end_capture(capture_id)
