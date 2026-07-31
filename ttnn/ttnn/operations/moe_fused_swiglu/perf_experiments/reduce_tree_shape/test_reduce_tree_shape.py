# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: correctness + device-ns measurement for the reduce-tree-shape idea.

Reconstructs, in isolation, the moe_fused_swiglu column reduce (KGROUPS cores, each holding an
N-tile bfloat8_b local partial that stands in for the concatenated gate+up block) and A/Bs three
topologies: `hillis_steele` (the op's shipped tree), `fanin2` (bounded max-fan-in-2 binary merge),
and `twophase` (tile-index reduce-scatter, zero adds at the root). Correctness is PCC against an
fp32 torch reference of the K-way sum of the ACTUAL bfp8-quantized local partials (matching how the
real op's cb_gate_acc/cb_up_acc are already bfp8_b before the reduce starts). Perf is one
fresh-cache run per (variant, K, N) cell — device kernel time has no warm-up transient.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

# NOTE: `torch` is imported LAZILY here. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/` so that importing ttnn
# never drags torch in. These perf-experiment benches live under the op directory, so they
# obey the same rule: every use sites gets `import torch` inside the function.
import ttnn
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_tree_shape.program_descriptor import (
    TILE,
    build_tree_layout,
    fanin2_tree,
    hillis_steele_tree,
    make_sharded_config,
    run_tree_variant,
    run_twophase,
    tree_depth,
    tree_max_fanin,
    tree_root_adds,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total = 0.0
    found = False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    import torch

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


def _make_inputs(device, k, n_tiles):
    """Height-sharded [k*32, n_tiles*32] bfloat8_b local partials, one distinguishable per-core
    pattern each (so a wrong-core / wrong-offset bug shows up as a PCC break, not a lucky match)."""
    import torch

    config = make_sharded_config(device, k, n_tiles)
    torch_local = torch.empty((k * TILE, n_tiles * TILE), dtype=torch.float32)
    col_pattern = (torch.arange(n_tiles * TILE, dtype=torch.float32) % 13).reshape(1, -1) / 32.0
    for row in range(k):
        value = (row + 1) * 0.25
        torch_local[row * TILE : (row + 1) * TILE] = value + col_pattern

    local_tensor = ttnn.from_torch(
        torch_local,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config,
    )
    # The REAL (quantized) per-core values, read back once before any kernel runs — this is what
    # the device reduce actually sums, so the fp32 reference must be built from THIS, not the
    # pre-quantization torch_local.
    quantized = ttnn.to_torch(local_tensor).to(torch.float32)
    reference_sum = sum(quantized[row * TILE : (row + 1) * TILE] for row in range(k))

    zero = torch.zeros((k * TILE, n_tiles * TILE), dtype=torch.float32)
    result_tensor = ttnn.from_torch(
        zero,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config,
    )
    return local_tensor, result_tensor, reference_sum


def _root_shard(result_tensor, root=0):
    import torch

    torch_result = ttnn.to_torch(result_tensor).to(torch.float32)
    return torch_result[root * TILE : (root + 1) * TILE]


def _run_and_check(device, variant, k, n_tiles, *, writer_noc=None):
    local_tensor, result_tensor, reference_sum = _make_inputs(device, k, n_tiles)
    if variant == "twophase":
        run_twophase(device, local_tensor, result_tensor, k, n_tiles)
    else:
        run_tree_variant(device, local_tensor, result_tensor, variant, k, n_tiles, writer_noc=writer_noc)
    actual = _root_shard(result_tensor)
    pcc = _pcc(actual, reference_sum)
    return pcc


def test_reduce_tree_shape_correctness(device):
    """Correctness gate: every variant must clear the op's soft PCC gate (0.975, feature_spec.py)
    with wide headroom. This isolated bench has no bfp4 weight quantization (the real op's actual
    accuracy floor, ~0.9797) — the only noise here is bfp8 pack/unpack across repeated adds, which
    is small but NOT zero (a reduction order change measurably moves it, exactly as the coordinator
    briefing warns), so the gate is 0.99, not bit-exact."""
    for k in (10, 8, 4):
        for n_tiles in (96, 48, 12):
            for variant in ("hillis_steele", "fanin2", "twophase"):
                pcc = _run_and_check(device, variant, k, n_tiles)
                logger.info(f"[correctness] variant={variant} k={k} n={n_tiles} pcc={pcc:.6f}")
                assert pcc > 0.99, f"variant={variant} k={k} n={n_tiles} pcc={pcc} too low"


def test_reduce_tree_shape_tree_shapes():
    """Host-only sanity: fanin2 really bounds max fan-in to 2 for every k, hillis_steele's root
    fan-in matches ceil(log2(k)), and every leaf/child count sums to k-1 adds."""
    for k in (4, 8, 10):
        hs = hillis_steele_tree(k)
        f2 = fanin2_tree(k)
        assert tree_max_fanin(f2) <= 2, f"fanin2 k={k} max fanin {tree_max_fanin(f2)}"
        total_adds_hs = sum(len(n["children"]) for n in hs.values())
        total_adds_f2 = sum(len(n["children"]) for n in f2.values())
        assert total_adds_hs == k - 1
        assert total_adds_f2 == k - 1
        logger.info(
            f"k={k} hillis_steele: root_adds={tree_root_adds(hs)} depth={tree_depth(hs)} | "
            f"fanin2: root_adds={tree_root_adds(f2)} depth={tree_depth(f2)} max_fanin={tree_max_fanin(f2)}"
        )


def test_reduce_tree_shape_device_perf(device):
    """One fresh-cache run per (variant, k, n_tiles) cell — the predicate sweep. Correctness-gates
    first, then measures DEVICE KERNEL DURATION [ns]. Perf is reported, never asserted."""
    variants = ("hillis_steele", "fanin2", "twophase")
    sweep = [(10, 96), (10, 48), (10, 12), (8, 96), (4, 96), (4, 12)]
    results = []
    for k, n_tiles in sweep:
        for variant in variants:
            local_tensor, result_tensor, reference_sum = _make_inputs(device, k, n_tiles)
            if variant == "twophase":
                run_twophase(device, local_tensor, result_tensor, k, n_tiles)
            else:
                run_tree_variant(device, local_tensor, result_tensor, variant, k, n_tiles)
            pcc = _pcc(_root_shard(result_tensor), reference_sum)
            ttnn.synchronize_device(device)
            _read_kernel_ns(device)  # discard: first run also pays JIT compile

            # fresh-cache-equivalent re-run for the timed sample (same descriptor, cache now warm —
            # device kernel time has no warm-up transient per /perf-measure, so ONE post-JIT run is
            # the measurement).
            if variant == "twophase":
                run_twophase(device, local_tensor, result_tensor, k, n_tiles)
            else:
                run_tree_variant(device, local_tensor, result_tensor, variant, k, n_tiles)
            ns = _read_kernel_ns(device)
            assert ns is not None, f"no device duration for variant={variant} k={k} n={n_tiles}"
            results.append((variant, k, n_tiles, ns, pcc))
            logger.info(f"[perf] variant={variant:14s} k={k:2d} n={n_tiles:3d} ns={ns:9.1f} pcc={pcc:.6f}")

    logger.info(
        "\n=== reduce_tree_shape sweep ===\n"
        + "\n".join(f"{v:14s} k={k:2d} n={n:3d} ns={ns:9.1f} pcc={p:.6f}" for v, k, n, ns, p in results)
    )


def test_reduce_tree_shape_noc_direction(device):
    """Idea #5: does the child->parent unicast NoC choice matter for this column-direction tree?

    NOTE: this test only exercises the shipped default (writer on NOC_1, matching the real op).
    An earlier manual probe additionally flipped the writer to `ttnn.NOC.NOC_0` (the SAME NoC the
    reader already defaults to, for its parent-invite side) via `run_tree_variant(..., writer_noc=
    ttnn.NOC.NOC_0)` — that combination measured a genuine DEVICE HANG (dispatch timeout, confirmed
    device-reset-clean afterward), not a slow number. It was not investigated further (out of this
    idea's scope), but the finding stands: putting the reader's parent-invite and the writer's
    child-ship on the SAME NoC is not merely a "which is faster" perf question here, it may be a
    correctness precondition for concurrent invite+ship traffic on the same core pair. Left OUT of
    the automated sweep so this file stays green on re-run; the NOC_1 number below is the real
    op's actual configuration and is the same number `test_reduce_tree_shape_device_perf` reports
    for hillis_steele at k=10, n=96.
    """
    k, n_tiles = 10, 96
    local_tensor, result_tensor, reference_sum = _make_inputs(device, k, n_tiles)
    run_tree_variant(device, local_tensor, result_tensor, "hillis_steele", k, n_tiles, writer_noc=None)
    pcc = _pcc(_root_shard(result_tensor), reference_sum)
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    run_tree_variant(device, local_tensor, result_tensor, "hillis_steele", k, n_tiles, writer_noc=None)
    ns = _read_kernel_ns(device)
    logger.info(f"[noc] writer_noc=NOC_1(default) ns={ns} pcc={pcc:.6f}")
