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

import contextlib
import inspect
import os

import pytest

_ACTIVE = os.environ.get("UP_FRONT_COLLECT") == "1"
_WORKERS = int(os.environ.get("UP_FRONT_COLLECT_WORKERS", "0"))  # 0 => hardware_concurrency
_DEVICE_ID = int(os.environ.get("UP_FRONT_COLLECT_DEVICE_ID", "0"))
# UP_FRONT_REAL_ALLOC=1: collect with REAL buffer addresses (dispatch still blocked) instead of
# addr-0 mocking, so address-baked / address-branched kernels (pool reader, move fwd/bwd) warm.
# Costs real device memory (~the real run's peak) — only use when the model fits.
_REAL_ALLOC = os.environ.get("UP_FRONT_REAL_ALLOC") == "1"
# UP_FRONT_FAST_COLLECT=1 (default): in the collect window (results thrown away under NO_DISPATCH),
# replace expensive HOST-side torch work — torch.randn -> zeros, the torch *reference* conv
# (torch.nn.functional.conv2d) -> shape-correct zeros, comp_pcc -> no-op. Values are irrelevant
# under NO_DISPATCH and collected programs depend only on ttnn op shapes/config, so this does NOT
# change what is collected — it stops pass 1 paying for the golden reference + PCC + RNG.
_FAST_COLLECT = os.environ.get("UP_FRONT_FAST_COLLECT", "1") == "1"

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


@contextlib.contextmanager
def _cheap_host_ops():
    """Swap expensive host-side torch work for cheap shape-correct stand-ins during collect.

    Only ttnn op shapes/config determine the collected programs, so zeroing the inputs, the
    torch reference conv, and the PCC check does not change WHAT is collected — it just removes
    the wasted golden-reference + PCC + RNG cost from the throwaway collect pass.
    """
    import sys

    import torch
    import torch.nn.functional as F

    real_randn = torch.randn
    real_rand = torch.rand
    real_conv2d = F.conv2d
    real_layer_norm = F.layer_norm

    def _fast_randn(*size, **kw):
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = tuple(size[0])
        kw.pop("generator", None)
        return torch.zeros(*size, **kw)

    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def _fast_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        sH, sW = _pair(stride)
        pH, pW = _pair(padding)
        dH, dW = _pair(dilation)
        N, C_out = input.shape[0], weight.shape[0]
        H, W = input.shape[-2], input.shape[-1]
        kH, kW = weight.shape[-2], weight.shape[-1]
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return input.new_zeros((N, C_out, H_out, W_out))

    def _fast_layer_norm(input, *a, **k):
        return torch.zeros_like(input)  # layer_norm / rms_norm output shape == input shape

    # Return a high-but-not-exactly-1 PCC so `assert passing` passes while the common
    # `if pcc == 1: assert_equal(...)` branch is skipped (that would re-touch full tensors).
    pcc_saved = []
    for _modname in ("tests.ttnn.utils_for_testing", "models.common.utility_functions"):
        _mod = sys.modules.get(_modname)
        if _mod is not None and hasattr(_mod, "comp_pcc"):
            pcc_saved.append((_mod, _mod.comp_pcc))
            _mod.comp_pcc = lambda *a, **k: (True, 0.999999)

    torch.randn = _fast_randn
    torch.rand = _fast_randn
    F.conv2d = _fast_conv2d
    F.layer_norm = _fast_layer_norm
    try:
        yield
    finally:
        torch.randn = real_randn
        torch.rand = real_rand
        F.conv2d = real_conv2d
        F.layer_norm = real_layer_norm
        for _mod, _orig in pcc_saved:
            _mod.comp_pcc = _orig


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
    ttnn.graph.up_front_begin_collect(clear=False, real_alloc=_REAL_ALLOC)  # accumulate; wraps ONLY the body
    try:
        if _FAST_COLLECT:
            with _cheap_host_ops():
                return (yield)  # body runs with cheap host stand-ins; ttnn ops still stash
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
