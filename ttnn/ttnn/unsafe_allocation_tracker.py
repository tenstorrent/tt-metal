# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tracks device buffer allocations made while traces are active, to catch
buffers that would be silently corrupted on trace replay.

Controlled by environment variables:
  TT_METAL_TRACE_ALLOC_TRACKING=1   Enable tracking (error on untracked
                                     buffers surviving until execute_trace).
  TT_METAL_TRACE_ALLOC_TRACEBACKS=1 Also capture Python call stacks via
                                     sys.setprofile for richer error messages.

See also: ttnn.corruptible_allocation_scope, ttnn.execute_trace.
"""

from __future__ import annotations

import gc
import os
import sys
import traceback
import threading
from typing import ClassVar


class UnsafeAllocationTracker:
    """Per-device tracker for allocations made while traces are active."""

    _profile_hook_installed: ClassVar[bool] = False
    _tls: ClassVar[threading.local] = threading.local()
    _tracebacks: ClassVar[dict[int, str]] = {}

    _env_tracebacks: ClassVar[bool] = os.environ.get("TT_METAL_TRACE_ALLOC_TRACEBACKS") is not None

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

    @classmethod
    def _profile_hook(cls, frame, event, arg):
        """sys.setprofile callback. On c_return, drain pending IDs and snapshot the caller's stack."""
        if event != "c_return":
            return
        if getattr(cls._tls, "draining", False):
            return
        cls._tls.draining = True
        try:
            from ttnn._ttnn.operations.trace import drain_pending_traceback_ids

            pending = drain_pending_traceback_ids()
            if not pending:
                return
            tb = "".join(traceback.format_stack(frame))
            for buf_id in pending:
                cls._tracebacks[buf_id] = tb
        finally:
            cls._tls.draining = False

    @classmethod
    def _install_profile_hook(cls):
        if cls._profile_hook_installed:
            return
        sys.setprofile(cls._profile_hook)
        threading.setprofile(cls._profile_hook)
        cls._profile_hook_installed = True

    def verify_before_replay(self) -> None:
        """
        Call before execute_trace. Triggers GC, then checks for unsafe
        buffers still alive. Raises RuntimeError with details if any are found.
        """
        from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

        gc.collect()

        live_unsafe = get_unsafe_tracked_ids(self.mesh_device)
        if not live_unsafe:
            return

        parts = [
            f"Found {len(live_unsafe)} device buffer(s) still alive before "
            f"trace replay. These will be corrupted on replay.\n"
        ]
        for buf_id in sorted(live_unsafe):
            tb = self._tracebacks.get(buf_id)
            if tb:
                parts.append(f"\nBuffer {buf_id} allocated at:\n{tb}")
            else:
                parts.append(
                    f"\nBuffer {buf_id} (no traceback captured; " f"set TT_METAL_TRACE_ALLOC_TRACEBACKS=1 for details)"
                )
        parts.append(
            "\nUse corruptible_allocation_scope() for acknowledged-corruptible "
            "tensors, or ensure temporary tensors are freed before replay."
        )
        raise RuntimeError("".join(parts))


if UnsafeAllocationTracker._env_tracebacks:
    UnsafeAllocationTracker._install_profile_hook()
