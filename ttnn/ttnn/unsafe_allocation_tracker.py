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

        Reports C++ live-Tensor registry info, shared_ptr refcounts, and
        Python-side referrers.
        """
        import ttnn as _ttnn
        from ttnn._ttnn.operations.trace import (
            get_unsafe_tracked_ids,
            get_unsafe_buffer_refcounts,
            get_live_tensor_buffer_info,
        )

        _ttnn.synchronize_device(self.mesh_device)
        gc.collect()

        live_unsafe = get_unsafe_tracked_ids(self.mesh_device)
        if not live_unsafe:
            return

        refcounts = get_unsafe_buffer_refcounts(self.mesh_device)
        live_tensor_info = get_live_tensor_buffer_info(live_unsafe)

        parts = [
            f"Found {len(live_unsafe)} device buffer(s) still alive before "
            f"trace replay. These will be corrupted on replay.\n"
        ]

        parts.append(
            f"\n--- C++ live-Tensor registry ({len(live_tensor_info)} Tensor(s) holding tracked buffers) ---\n"
        )
        info_by_buf: dict[int, list] = {}
        for info in live_tensor_info:
            info_by_buf.setdefault(info["buffer_unique_id"], []).append(info)

        for buf_id in sorted(live_unsafe):
            rc = refcounts.get(buf_id)
            rc_str = f"  (Buffer shared_ptr refcount = {rc})" if rc is not None else ""
            holders = info_by_buf.get(buf_id, [])
            if holders:
                for h in holders:
                    parts.append(
                        f"Buffer {buf_id}{rc_str}: held by C++ Tensor id={h['tensor_id']}, "
                        f"tensor_attrs.use_count={h['tensor_attrs_refcount']}, "
                        f"mesh_buffer.use_count={h['mesh_buffer_refcount']}"
                    )
            else:
                parts.append(f"Buffer {buf_id}{rc_str}: NO live C++ Tensor found (orphaned MeshBuffer?)")

            tb = self._tracebacks.get(buf_id)
            if tb:
                parts.append(f"  allocated at:\n{tb}")

        orphaned_bufs = [bid for bid in sorted(live_unsafe) if bid not in info_by_buf]
        if orphaned_bufs:
            parts.append(
                f"\n{len(orphaned_bufs)} buffer(s) with no live C++ Tensor holder " f"(first 10): {orphaned_bufs[:10]}"
            )

        parts.append("\n--- Python referrer analysis ---\n")
        try:
            parts.append(self._find_python_referrers(live_unsafe))
        except Exception as e:
            parts.append(f"(referrer analysis failed: {type(e).__name__}: {e})")

        parts.append(
            "\nUse corruptible_allocation_scope() for acknowledged-corruptible "
            "tensors, or ensure temporary tensors are freed before replay."
        )
        raise RuntimeError("".join(parts))

    @staticmethod
    def _find_python_referrers(live_unsafe: set[int]) -> str:
        """Walk all live Python stack frames to find ttnn.Tensor instances
        whose backing buffer unique_id is in *live_unsafe*.

        Recurses into frame locals, their list/dict/tuple children, and
        object __dict__ attributes (two levels deep) to find tensors held
        by e.g. self.some_list[i].
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
                for k, v in val.items():
                    _scan_value(v, loc, f"{path}[{k!r}]", depth - 1)
            elif hasattr(val, "__dict__") and not isinstance(val, type):
                stats["objects_traversed"] += 1
                try:
                    obj_dict = val.__dict__
                except Exception:
                    return
                for attr_name, attr_val in obj_dict.items():
                    if attr_name.startswith("__"):
                        continue
                    _scan_value(attr_val, loc, f"{path}.{attr_name}", depth - 1)

        for thread_id, frame in sys._current_frames().items():
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
                    _scan_value(val, loc, name, depth=5)
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
                key = (loc, path)
                if key in seen:
                    continue
                seen.add(key)
                shape = ""
                try:
                    shape = f", shape={tensor.shape}"
                except Exception:
                    pass
                lines.append(f"  '{path}'{shape}")
                lines.append(f"    at {loc}")

        unmatched = live_unsafe - set(found)
        if unmatched:
            lines.append(f"\n{len(unmatched)} buffer(s) not found in any Python stack frame.")
        return "\n".join(lines)


if UnsafeAllocationTracker._env_tracebacks:
    UnsafeAllocationTracker._install_profile_hook()
