# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""General, eval-independent up-front precompile collector for ANY ttnn pytest suite.

Phase 2 of the "marriage": drive mechanism #1's op-agnostic C++ collector
(``ttnn.graph.up_front_*``) across a whole pytest session via the call-phase
hookwrapper, so EVERY op of EVERY test body — multi-op bodies and C++
ProgramDescriptor-migrated ops alike — is stashed and deduped, then the distinct
set is JIT-compiled once, in parallel, at session end (warming the on-disk
``TT_METAL_CACHE``).

Why a hookwrapper (not eval's re-invocation): pytest runs each body through its
*own* fixture machinery, so this works for arbitrary tests without knowing their
signatures. ``begin_collect(clear=False)`` wraps ONLY the call phase, so
NO_DISPATCH never spans fixture setup/teardown (the device is created for real in
setup); ``clear=False`` makes the collector ACCUMULATE across tests.

TWO PASSES (required): under NO_DISPATCH each body runs "neutered" (mocked addr-0
buffers, nothing dispatched), so its asserts fail and are swallowed — pass 1 only
collects + compiles. Re-run the SAME suite over the SAME on-disk cache for the
real, now-warm results:

    # pass 1 — collect every op + parallel-compile the distinct set (warms the cache)
    UP_FRONT_COLLECT=1 TT_METAL_CACHE=/tmp/c  pytest -p up_front_collect_plugin <tests>
    # pass 2 — real run, warm (plugin off)
    TT_METAL_CACHE=/tmp/c  pytest <tests>

Knobs:  UP_FRONT_COLLECT=1 (enable; no-op otherwise) ·
        UP_FRONT_COLLECT_WORKERS=N (0 => hardware_concurrency) ·
        UP_FRONT_COLLECT_DEVICE_ID=N (device for the session-end compile, default 0)

Limitations (see MODEL_PRECOMPILE_DESIGN.md §8.1):
  * Tests that drive trace/graph capture are SKIPPED — NO_DISPATCH blocks the real
    dispatch + buffer allocation that recording needs. They cold-compile in pass 2.
  * A body that reads a tensor back / branches on values mid-body breaks on addr-0:
    we capture the ops before that point; the rest cold-compile in pass 2.
  * Ops that pick their program from live allocator state see empty L1 under
    NO_DISPATCH and may collect a different variant than the real run uses.
"""

from __future__ import annotations

import inspect
import os

import pytest

_ACTIVE = os.environ.get("UP_FRONT_COLLECT") == "1"
_WORKERS = int(os.environ.get("UP_FRONT_COLLECT_WORKERS", "0"))  # 0 => hardware_concurrency
_DEVICE_ID = int(os.environ.get("UP_FRONT_COLLECT_DEVICE_ID", "0"))

# Bodies that drive trace/graph capture can't run under NO_DISPATCH (it blocks the
# dispatch + alloc that recording needs). Heuristic: scan the test function source.
_CAPTURE_MARKERS = (
    "begin_trace_capture",
    "capture_trace",
    "execute_trace",
    "begin_graph_capture",
    "begin_mesh_trace_capture",
)

_stats = {"bodies": 0, "swallowed": 0, "skipped_capture": 0}


def _drives_capture(item) -> bool:
    fn = getattr(item, "function", None)
    if fn is None:
        return False
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return any(m in src for m in _CAPTURE_MARKERS)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Run each body under one NO_DISPATCH collect window (call phase only)."""
    if not _ACTIVE:
        return (yield)

    import ttnn

    if _drives_capture(item):
        _stats["skipped_capture"] += 1
        return (yield)  # let it run normally; it cold-compiles in pass 2

    _stats["bodies"] += 1
    ttnn.graph.up_front_begin_collect(clear=False)  # accumulate; NO_DISPATCH wraps ONLY the body
    try:
        return (yield)  # pytest runs the body with its real fixtures; ops stash into the collector
    except Exception:
        # Expected under NO_DISPATCH: a readback/assert on an addr-0 output. The
        # program was already stashed by the funnel before the failure. Swallow so
        # pass 1 stays green (its results are meaningless — pass 2 is the real run).
        _stats["swallowed"] += 1
        return None
    finally:
        ttnn.graph.up_front_end_collect()


def pytest_sessionstart(session):
    if not _ACTIVE:
        return
    import ttnn

    ttnn.graph.up_front_clear()  # clean slate before accumulating across the session


def pytest_sessionfinish(session, exitstatus):
    if not _ACTIVE:
        return
    import ttnn

    n_unique = ttnn.graph.up_front_num_unique()
    n_collected = ttnn.graph.up_front_num_collected()
    print(
        f"\nUP_FRONT_COLLECT: {n_collected} ops stashed across {_stats['bodies']} bodies "
        f"-> {n_unique} unique programs "
        f"(swallowed {_stats['swallowed']}, skipped-capture {_stats['skipped_capture']})",
        flush=True,
    )
    if n_unique == 0:
        print("UP_FRONT_COLLECT: nothing to compile", flush=True)
        return

    device = None
    try:
        device = ttnn.CreateDevice(device_id=_DEVICE_ID)
        try:
            device.enable_program_cache()
        except Exception:
            pass
        n_prog, n_err, used, wall = ttnn.graph.up_front_compile(device, _WORKERS, True)
        print(
            f"UP_FRONT_COLLECT: compiled {n_prog} programs in {wall:.1f}s "
            f"(workers={used}, errors={n_err}) -> on-disk JIT cache warm",
            flush=True,
        )
    except Exception as e:  # best-effort: never break the session
        print(f"UP_FRONT_COLLECT: compile skipped (best-effort) — {e!r}", flush=True)
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass
