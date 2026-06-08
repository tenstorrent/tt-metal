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
# UP_FRONT_META_COLLECT=1 (opt-in): the GENERIC version of fast-collect. Instead of per-op torch
# swaps (randn/conv2d/layer_norm/...), run the throwaway body on the torch META device: tensors
# carry shape+dtype but NO storage, and the dispatcher propagates shapes through ANY torch op with
# no compute and no allocation. This is the host-side analogue of the mock ttnn device — it removes
# the host memory + compute that OOM'd full-suite collects. The one ttnn<->torch boundary is bridged
# by allocate_tensor_on_device (from_torch) + a meta stand-in (to_torch), valid because program
# capture depends only on shapes/layout, not values. Falls back to real per-input on any edge case.
_META_COLLECT = os.environ.get("UP_FRONT_META_COLLECT") == "1"
# UP_FRONT_COLLECT_NO_COMPILE=1: collect/dedup only, skip the session-end parallel compile. Used to
# isolate the body-run cost from the compile floor when benchmarking (e.g., fast vs meta collect).
_NO_COMPILE = os.environ.get("UP_FRONT_COLLECT_NO_COMPILE") == "1"

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


_meta_stats = {"from_torch_alloc": 0, "from_torch_fallback": 0, "to_torch_meta": 0}


@contextlib.contextmanager
def _meta_host_ops():
    """Generic fake-torch collect: run the throwaway body on META tensors (no storage, no compute).

    Replaces the per-op swap list in _cheap_host_ops. Forces torch tensor creation onto the meta
    device so inputs AND every torch reference op become storage-less shape propagation (the
    dispatcher handles all ops generically — no per-op patching, no allocation -> no OOM). Bridges
    the ttnn boundary: from_torch(meta) -> allocate_tensor_on_device (shape-only, no host copy),
    to_torch -> meta stand-in. comp_pcc is a no-op. Any path that can't be meta'd falls back to a
    real shape-correct zeros for that one tensor, so the body still reaches the ttnn op to capture it.
    """
    import sys

    import torch
    import ttnn

    real = {
        n: getattr(torch, n)
        for n in (
            "randn",
            "rand",
            "zeros",
            "ones",
            "empty",
            "full",
            "arange",
            "randint",
            "tensor",
            "as_tensor",
            "eye",
            "linspace",
            "zeros_like",
            "ones_like",
            "empty_like",
            "randn_like",
            "rand_like",
        )
        if hasattr(torch, n)
    }
    real_from_torch = ttnn.from_torch
    real_to_torch = ttnn.to_torch
    real_item = torch.Tensor.item
    real_tolist = torch.Tensor.tolist

    _T2T = {
        torch.float32: ttnn.float32,
        torch.float64: ttnn.float32,
        torch.bfloat16: ttnn.bfloat16,
        torch.float16: ttnn.bfloat16,
        torch.int32: ttnn.uint32,
        torch.int64: ttnn.uint32,
        torch.int16: ttnn.uint16,
        torch.uint8: ttnn.uint8,
        torch.bool: ttnn.uint8,
    }

    def _mk(orig):
        def wrapped(*a, **kw):
            kw.setdefault("device", "meta")
            kw.pop("generator", None)
            try:
                return orig(*a, **kw)
            except (TypeError, RuntimeError):
                kw.pop("device", None)  # creator doesn't accept device= (rare) -> real
                return orig(*a, **kw)

        return wrapped

    def _meta_from_torch(
        tensor, dtype=None, *, spec=None, layout=None, device=None, memory_config=None, mesh_mapper=None, **kw
    ):
        is_meta = tensor is not None and getattr(tensor, "is_meta", False)
        if is_meta and device is not None:
            try:
                if spec is not None:
                    dt, lay, mc = spec.dtype, spec.layout, spec.memory_config
                    shp = ttnn.Shape(tuple(spec.shape))
                else:
                    dt = dtype or _T2T.get(tensor.dtype, ttnn.bfloat16)
                    lay = layout or ttnn.ROW_MAJOR_LAYOUT
                    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                    shp = ttnn.Shape(tuple(tensor.shape))
                t = ttnn.allocate_tensor_on_device(shp, dt, lay, device, mc)
                _meta_stats["from_torch_alloc"] += 1
                return t
            except Exception:
                pass  # fall through to materialize
        if is_meta:
            # device=None (host tensor) or alloc failed -> materialize a real shape-correct zeros
            _meta_stats["from_torch_fallback"] += 1
            tensor = real["zeros"](tuple(tensor.shape), dtype=tensor.dtype)
        return real_from_torch(
            tensor,
            dtype,
            spec=spec,
            layout=layout,
            device=device,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
            **kw,
        )

    def _meta_to_torch(tensor, *a, **kw):
        try:
            _meta_stats["to_torch_meta"] += 1
            return real["zeros"](tuple(tensor.shape), device="meta")
        except Exception:
            return real_to_torch(tensor, *a, **kw)

    pcc_saved = []
    for _modname in ("tests.ttnn.utils_for_testing", "models.common.utility_functions"):
        _mod = sys.modules.get(_modname)
        if _mod is not None and hasattr(_mod, "comp_pcc"):
            pcc_saved.append((_mod, _mod.comp_pcc))
            _mod.comp_pcc = lambda *a, **k: (True, 0.999999)

    # .item()/.tolist() RAISE on meta tensors. A body that reads a scalar out of a tensor to drive
    # HOST-side control flow before its ttnn ops (e.g. test_layer_norm_with_padding does
    # `torch.randint(1, w+1, (1,)).item()` on line 1) would abort under meta collect BEFORE any ttnn
    # op is stashed -> the whole test (and its fill_pad/layer_norm kernels) goes uncollected. The
    # extracted value only feeds input *data*, never the ttnn program's shape/config (which is all
    # that determines the collected kernels), so a deterministic stand-in is safe and lets the body
    # reach its ops. Only meta tensors are intercepted; real tensors keep exact semantics.
    def _meta_item(self):
        if getattr(self, "is_meta", False):
            return 1
        return real_item(self)

    def _meta_tolist(self):
        if getattr(self, "is_meta", False):
            return real_tolist(real["zeros"](tuple(self.shape)))
        return real_tolist(self)

    for name, orig in real.items():
        setattr(torch, name, _mk(orig))
    ttnn.from_torch = _meta_from_torch
    ttnn.to_torch = _meta_to_torch
    torch.Tensor.item = _meta_item
    torch.Tensor.tolist = _meta_tolist
    try:
        yield
    finally:
        for name, orig in real.items():
            setattr(torch, name, orig)
        ttnn.from_torch = real_from_torch
        ttnn.to_torch = real_to_torch
        torch.Tensor.item = real_item
        torch.Tensor.tolist = real_tolist
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
    # UP_FRONT_LOG_SWALLOWED=1: per-body diagnostic — which tests threw under NO_DISPATCH and
    # how many ops each stashed before the throw (reveals coverage: a body that throws BEFORE
    # stashing its op is a real miss; one that throws on the addr-0 readback after stashing is fine).
    _log = os.environ.get("UP_FRONT_LOG_SWALLOWED")
    n_before = ttnn.graph.up_front_num_collected() if _log else 0
    swallowed = False
    exc_info = ""
    ttnn.graph.up_front_begin_collect(clear=False, real_alloc=_REAL_ALLOC)  # accumulate; wraps ONLY the body
    try:
        if _META_COLLECT:
            with _meta_host_ops():
                return (yield)  # body runs on meta (no storage/compute); ttnn ops still stash
        if _FAST_COLLECT:
            with _cheap_host_ops():
                return (yield)  # body runs with cheap host stand-ins; ttnn ops still stash
        return (yield)  # pytest runs the body with its real fixtures; ops stash into the collector
    except Exception as e:
        # Expected under NO_DISPATCH: a readback/assert on an addr-0 output. The
        # program was already stashed by the funnel before the failure. Swallow so
        # pass 1 stays green (its results are meaningless — pass 2 is the real run).
        _stats["swallowed"] += 1
        swallowed = True
        exc_info = f"{type(e).__name__}: {str(e)[:140]}"
        return None
    finally:
        ttnn.graph.up_front_end_collect()
        if _log:
            n_after = ttnn.graph.up_front_num_collected()
            print(
                f"UP_FRONT_BODY: {'SWALLOWED' if swallowed else 'ok       '} "
                f"stashed={n_after - n_before:<3} {item.nodeid}" + (f"  -> {exc_info}" if swallowed else ""),
                flush=True,
            )


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
    if _META_COLLECT:
        print(
            f"UP_FRONT_COLLECT: meta-collect boundary — from_torch alloc={_meta_stats['from_torch_alloc']} "
            f"fallback={_meta_stats['from_torch_fallback']}, to_torch meta={_meta_stats['to_torch_meta']}",
            flush=True,
        )
    if n_unique == 0:
        print("UP_FRONT_COLLECT: nothing to compile", flush=True)
        return
    if _NO_COMPILE:
        print(f"UP_FRONT_COLLECT: NO_COMPILE set — collected {n_unique}, skipping compile", flush=True)
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
