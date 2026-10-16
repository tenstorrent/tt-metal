# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tracks device buffer allocations made while traces are active, to catch
buffers that would be silently corrupted on trace replay.

Controlled by environment variables:
  TT_METAL_TRACE_ALLOC_TRACKING=1       Enable accounting and fail when unsafe
                                         buffers survive until execute_trace.
  TT_METAL_TRACE_ALLOC_TRACEBACKS=1     With accounting enabled, capture Python
                                         allocation stacks and analyze referrers.
  TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=N Maximum referrer traversal depth
                                         (default: 10).

Set these before importing ttnn; they are read once at process startup.

See also: acknowledge_corruptible, corruptible_allocation_scope, ttnn.execute_trace.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import warnings
from typing import ClassVar

from ttnn._ttnn.operations.trace import (
    get_all_unsafe_tracked_ids,
    get_unsafe_tracked_ids,
    remove_unsafe_tracked_id,
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
# reading the environment independently, so TTNN and Metal cannot disagree. Referrer depth is Python-only.
TRACE_ALLOC_TRACKING = trace_allocation_tracking_enabled()
TRACE_ALLOC_DIAGNOSTICS = trace_allocation_diagnostics_enabled()
TRACE_ALLOC_REFERRER_DEPTH = (
    _env_nonnegative_int("TT_METAL_TRACE_ALLOC_REFERRER_DEPTH", 10) if TRACE_ALLOC_DIAGNOSTICS else 10
)


if TRACE_ALLOC_TRACKING:
    from ttnn._ttnn.operations.trace import (
        pop_corruptible_allocation_scope as _pop_corruptible_allocation_scope,
        push_corruptible_allocation_scope as _push_corruptible_allocation_scope,
    )

    @contextlib.contextmanager
    def corruptible_allocation_scope(mesh_device):
        """Suppress accounting for intentionally corruptible allocations in this scope."""
        _push_corruptible_allocation_scope(mesh_device)
        try:
            yield
        finally:
            _pop_corruptible_allocation_scope(mesh_device)

else:

    @contextlib.contextmanager
    def corruptible_allocation_scope(mesh_device):
        """No-op when trace allocation tracking is disabled."""
        yield


class TraceAllocationTracker:
    """Per-device tracker for allocations made while traces are active."""

    _tracebacks: ClassVar[dict[int, str]] = {}

    _diagnostics_enabled: ClassVar[bool] = TRACE_ALLOC_DIAGNOSTICS
    _referrer_depth: ClassVar[int] = TRACE_ALLOC_REFERRER_DEPTH

    @classmethod
    def reconcile_tracebacks(cls) -> set[int]:
        """Drop tracebacks for buffers no longer marked unsafe by any allocator."""
        currently_unsafe = set(get_all_unsafe_tracked_ids())
        for buffer_unique_id in cls._tracebacks.keys() - currently_unsafe:
            cls._tracebacks.pop(buffer_unique_id)
        return currently_unsafe

    @classmethod
    def acknowledge_corruptible(cls, tensor) -> None:
        """
        Acknowledge that a tensor's backing buffer may intentionally be corrupted.
        This removes the buffer from trace-allocation tracking immediately.

        This is a no-op when trace allocation tracking is disabled.
        """
        if not TRACE_ALLOC_TRACKING:
            return

        import ttnn

        if not isinstance(tensor, ttnn.Tensor):
            raise TypeError(f"acknowledge_corruptible expects a ttnn.Tensor, got {type(tensor).__name__}")
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError("acknowledge_corruptible expects a tensor with device storage")
        if not tensor.is_allocated():
            raise ValueError("acknowledge_corruptible expected an allocated tensor")

        buf_id = tensor.buffer_unique_id()
        if buf_id is None:
            raise ValueError("acknowledge_corruptible expected a tensor with a valid device buffer_unique_id")

        remove_unsafe_tracked_id(tensor.device(), buf_id)
        cls._tracebacks.pop(buf_id, None)

    @classmethod
    def verify_before_replay(cls, mesh_device, trace_id) -> None:
        """
        Call before execute_trace. Checks for unsafe buffers and only triggers
        GC when the first check finds candidates, then checks again. Raises
        RuntimeError with details if any remain.

        Reports allocation context (op name + compile args) and Python-side referrers.
        """
        # get_unsafe_tracked_ids returns dict[int, str] mapping buffer_id -> allocation context
        live_unsafe_map = get_unsafe_tracked_ids(mesh_device, trace_id)
        if not live_unsafe_map:
            if cls._diagnostics_enabled:
                cls.reconcile_tracebacks()
            return

        # Unreachable tensors can be retained in Python reference cycles. Pay
        # for a full collection only on the exceptional path where C++ first
        # reports a live unsafe allocation, then re-query authoritative state.
        gc.collect()
        live_unsafe_map = get_unsafe_tracked_ids(mesh_device, trace_id)
        if cls._diagnostics_enabled:
            cls.reconcile_tracebacks()
        if not live_unsafe_map:
            return

        live_unsafe = set(live_unsafe_map.keys())

        parts = [
            f"Found {len(live_unsafe)} device buffer(s) still alive before "
            f"trace replay. These will be corrupted on replay.\n"
        ]

        for buf_id in sorted(live_unsafe):
            ctx = live_unsafe_map.get(buf_id, "")
            ctx_str = f" [op: {ctx}]" if ctx else ""
            parts.append(f"Buffer {buf_id}{ctx_str}\n")

            if cls._diagnostics_enabled:
                tb = cls._tracebacks.get(buf_id)
                if tb:
                    parts.append(f"  allocated at:\n{tb}")

        if cls._diagnostics_enabled:
            parts.append("\n--- Python referrer analysis ---\n")
            try:
                parts.append(cls._find_python_referrers(live_unsafe))
            except Exception as e:
                parts.append(f"(referrer analysis failed: {type(e).__name__}: {e})")

        parts.append(
            "\nUse ttnn.tools.trace_allocation_tracker.corruptible_allocation_scope() for acknowledged-corruptible "
            "tensors, or ensure temporary tensors are freed before replay."
        )
        raise RuntimeError("".join(parts))

    @staticmethod
    def _find_python_referrers(live_unsafe: set[int]) -> str:
        """Walk all live Python stack frames to find ttnn.Tensor instances
        whose backing buffer unique_id is in *live_unsafe*.

        Recurses into frame locals, their list/dict/tuple children, and
        object __dict__ attributes up to the configured referrer depth to find
        tensors held by e.g. self.some_list[i].
        """
        import ttnn

        stats = {
            "threads": 0,
            "frames": 0,
            "locals": 0,
            "tensors_seen": 0,
            "uid_ok": 0,
            "uid_none": 0,
            "uid_exc": 0,
            "matched": 0,
            "objects_traversed": 0,
        }
        all_uids_seen: list[int] = []
        # buf_id -> list of (location_str, path_str, tensor)
        found: dict[int, list[tuple[str, str, object]]] = {}
        visited: set[int] = set()

        def _get_uid(obj):
            if isinstance(obj, ttnn.Tensor):
                stats["tensors_seen"] += 1
                try:
                    uid = obj.buffer_unique_id()
                except Exception:
                    stats["uid_exc"] += 1
                    return None
                if uid is None:
                    stats["uid_none"] += 1
                else:
                    stats["uid_ok"] += 1
                    all_uids_seen.append(uid)
                return uid
            return None

        def _check(uid, loc, path, obj):
            if uid is not None and uid in live_unsafe:
                stats["matched"] += 1
                found.setdefault(uid, []).append((loc, path, obj))

        def _scan_value(val, loc, path, depth):
            """Recursively scan a value for tensors, up to *depth* levels."""
            obj_id = id(val)
            if obj_id in visited:
                return
            visited.add(obj_id)

            _check(_get_uid(val), loc, path, val)
            if depth <= 0:
                return

            if isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    _scan_value(item, loc, f"{path}[{idx}]", depth - 1)
            elif isinstance(val, dict):
                # Snapshot to avoid RuntimeError if dict mutates concurrently.
                # Some custom mapping types may still fail during snapshot; skip those.
                try:
                    items = list(val.items())
                except Exception:
                    return
                for k, v in items:
                    _scan_value(v, loc, f"{path}[{k!r}]", depth - 1)
            elif hasattr(val, "__dict__") and not isinstance(val, type):
                stats["objects_traversed"] += 1
                try:
                    obj_dict = val.__dict__
                except Exception:
                    return
                # Snapshot to avoid RuntimeError if object attrs change during scan.
                try:
                    attr_items = list(obj_dict.items())
                except Exception:
                    return
                for attr_name, attr_val in attr_items:
                    if attr_name.startswith("__"):
                        continue
                    _scan_value(attr_val, loc, f"{path}.{attr_name}", depth - 1)

        # Snapshot frame map to avoid surprises while iterating in highly concurrent runtimes.
        for thread_id, frame in list(sys._current_frames().items()):
            stats["threads"] += 1
            while frame is not None:
                stats["frames"] += 1
                loc = f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
                try:
                    local_items = list(frame.f_locals.items())
                except Exception:
                    frame = frame.f_back
                    continue
                for name, val in local_items:
                    stats["locals"] += 1
                    _scan_value(val, loc, name, depth=TraceAllocationTracker._referrer_depth)
                frame = frame.f_back

        lines: list[str] = [
            f"\n[scan stats] threads={stats['threads']}, frames={stats['frames']}, "
            f"locals={stats['locals']}, objects_traversed={stats['objects_traversed']}, "
            f"tensors_seen={stats['tensors_seen']}, "
            f"uid_ok={stats['uid_ok']}, uid_none={stats['uid_none']}, "
            f"uid_exc={stats['uid_exc']}, matched={stats['matched']}",
        ]
        if all_uids_seen:
            sample = sorted(set(all_uids_seen))[:30]
            lines.append(f"[buffer IDs seen ({len(set(all_uids_seen))} unique)] {sample}")
        lines.append(f"[looking for IDs ({len(live_unsafe)})] {sorted(live_unsafe)[:20]}...")

        for buf_id in sorted(found):
            refs = found[buf_id]
            lines.append(f"\nBuffer {buf_id}: found in {len(refs)} Python reference(s)")
            seen = set()
            for loc, path, tensor in refs:
                if path in seen:
                    continue
                seen.add(path)
                shape = ""
                try:
                    shape = f", shape={tensor.shape}"
                except Exception:
                    pass
                lines.append(f"  '{path}'{shape}")

        unmatched = live_unsafe - set(found)
        if unmatched:
            lines.append(f"\n{len(unmatched)} buffer(s) not found in any Python stack frame.")
            lines.append(
                "Hint: these may be program cache buffers (tracked by default; disable with"
                " TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1), or held deeper than the"
                f" referrer search depth (currently {TraceAllocationTracker._referrer_depth} levels)."
            )
        return "\n".join(lines)


def acknowledge_corruptible(tensor) -> None:
    """Acknowledge that a device tensor's buffer may intentionally be corrupted by trace replay."""
    TraceAllocationTracker.acknowledge_corruptible(tensor)
