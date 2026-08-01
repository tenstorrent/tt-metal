# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: correctness + device-ns for the reduce-scatter/distributed-SwiGLU idea.

Reconstructs ONE grid column of `moe_fused_swiglu`'s cross-K reduce + SwiGLU epilogue and A/Bs the
op's shipped binary tree against a two-phase reduce-scatter on two slice axes and two epilogue
placements. Correctness is PCC against an fp32 torch reference built from the ACTUAL bfp8-quantised
per-core partials (which is what the device sums). Perf is one post-JIT run per (variant, cell);
device kernel time has no warm-up transient, so a trial loop would just re-measure the same number.

  scripts/run_safe_pytest.sh --run-all <this file>
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

# NOTE: `torch` is imported LAZILY in every function. scripts/validate_no_global_torch_imports.py
# forbids a module-level torch import anywhere under ttnn/ttnn/, and these perf benches live under
# the op directory, so they obey the same rule.
import pytest
import ttnn
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_scatter_swiglu.program_descriptor import (
    TILE,
    build_layout,
    cb_bytes,
    hillis_steele_tree,
    run_variant,
    slice_plan,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# The focus shape's derived reduce geometry: emb 7168 / cap 5120 / count 256 gives KGROUPS = 10,
# HN_PAD = 6, M_BLOCK = 8 and runtime m_eff = 8, i.e. a 48-tile bfp8 gate block and a 48-tile up
# block per core, ten cores deep.
FOCUS = (10, 8, 6)  # (KGROUPS, m_eff, HN_PAD)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _pcc(a, b):
    import torch

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


def _make_inputs(device, layout):
    """Height-sharded bfp8 gate/up partials over the whole HGROUPS x KGROUPS grid, plus the zeroed h
    output. Returns ONE fp32 reference per column: SiLU(sum gate) * sum(up) built from the QUANTISED
    partials — that is what the device actually reduces."""
    import torch

    from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_scatter_swiglu.program_descriptor import (
        make_sharded_config,
    )

    k, t, ncols = layout.k, layout.t_tiles, layout.ncols
    n_shards = k * ncols
    config = make_sharded_config(device, k, t, ncols)

    # A per-core-distinct, per-tile-distinct pattern (so a wrong-core / wrong-offset / wrong-column
    # bug shows up as a PCC break rather than a lucky match); small magnitude so SiLU sits in its
    # curved region, where an epilogue bug cannot hide.
    #
    # The per-shard offset is a BOUNDED hash of the shard index, deliberately NOT proportional to it.
    # An offset that grows with the shard count (e.g. 0.11 * s over 110 shards) puts a DC term of
    # ~100 under a signal of ~0.3, and bfp8_b's 7-bit mantissa then quantises the signal away — the
    # 11-column run scored PCC 0.776 for the BASELINE on that pattern, i.e. it was the test input,
    # not the kernel. Verified separately (probes/probe_052.py) that the shard -> core map really is
    # ROW_MAJOR (shard index = row * ncols + col, so column c's root is shard c).
    wcol = (torch.arange(t * TILE, dtype=torch.float32) % 17) / 24.0 - 0.35
    hrow = (torch.arange(TILE, dtype=torch.float32) % 5) / 40.0
    gate = torch.empty((n_shards * TILE, t * TILE), dtype=torch.float32)
    up = torch.empty((n_shards * TILE, t * TILE), dtype=torch.float32)
    for s in range(n_shards):
        off_g = ((s * 7) % 23) / 23.0 - 0.5
        off_u = ((s * 11) % 19) / 19.0 - 0.5
        gate[s * TILE : (s + 1) * TILE] = off_g + wcol.reshape(1, -1) + hrow.reshape(-1, 1)
        up[s * TILE : (s + 1) * TILE] = off_u - wcol.reshape(1, -1) + hrow.reshape(-1, 1)

    gate_t = ttnn.from_torch(gate, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
    up_t = ttnn.from_torch(up, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
    h_t = ttnn.from_torch(
        torch.zeros((n_shards * TILE, t * TILE), dtype=torch.float32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config,
    )

    gq = ttnn.to_torch(gate_t).to(torch.float32)
    uq = ttnn.to_torch(up_t).to(torch.float32)
    references = []
    for col in range(ncols):
        gsum = sum(gq[layout.shard_index(col, r) * TILE : (layout.shard_index(col, r) + 1) * TILE] for r in range(k))
        usum = sum(uq[layout.shard_index(col, r) * TILE : (layout.shard_index(col, r) + 1) * TILE] for r in range(k))
        references.append(torch.nn.functional.silu(gsum) * usum)
    return gate_t, up_t, h_t, references


def _run_cell(device, variant, k, m_eff, hn_pad, *, measure=True, ncols=1):
    layout = build_layout(k, m_eff, hn_pad, ncols)
    gate_t, up_t, h_t, references = _make_inputs(device, layout)
    run_variant(device, gate_t, up_t, h_t, variant, layout)
    got = _h_torch(h_t)
    # EVERY column's root must be right, not just column 0.
    pcc = min(
        _pcc(got[layout.shard_index(col, 0) * TILE : (layout.shard_index(col, 0) + 1) * TILE], references[col])
        for col in range(ncols)
    )
    ns = None
    if measure:
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)  # discard: the first run also pays JIT compile
        run_variant(device, gate_t, up_t, h_t, variant, layout)
        ns = _read_kernel_ns(device)
    return pcc, ns, cb_bytes(variant, layout)


def _h_torch(h_t):
    import torch

    return ttnn.to_torch(h_t).to(torch.float32)


# ---------------------------------------------------------------------------
# Host-only sanity — no device
# ---------------------------------------------------------------------------


def test_geometry():
    """The baseline's root really does the fan-in the coordinator measured, and both slice plans
    really partition the block."""
    k, m_eff, hn_pad = FOCUS
    tree = hillis_steele_tree(k)
    root_fanin = len(tree[0]["children"])
    assert root_fanin == 4, root_fanin  # ceil(log2(10))
    assert sum(len(n["children"]) for n in tree.values()) == k - 1

    t = m_eff * hn_pad
    for kind in ("flat", "m", "ragged"):
        assigned, offsets, slice_pages = slice_plan(kind, k, m_eff, hn_pad)
        assert sum(assigned) == t
        assert offsets[0] == 0
        for a in assigned:
            if a:
                assert slice_pages % a == 0, (kind, assigned, slice_pages)
        logger.info(f"[plan] {kind}: assigned={assigned} offsets={offsets} slice_cb_pages={slice_pages}")
    # Baseline root tile-ops vs candidate per-core tile-ops, at the focus cell.
    base_ops = 2 * root_fanin * t + t  # gate adds + up adds + the SwiGLU multiply
    for kind in ("flat", "m", "ragged"):
        a = max(slice_plan(kind, k, m_eff, hn_pad)[0])
        logger.info(f"[ops] baseline root = {base_ops} tile-ops, {kind} worker = {2 * k * a + a} tile-ops")


# ---------------------------------------------------------------------------
# Device: correctness gate first (smallest cell first so a structural bug surfaces cheaply)
# ---------------------------------------------------------------------------


CORRECTNESS_CELLS = [
    (4, 2, 6),  # smallest: 12 tiles, 4 cores
    (10, 8, 6),  # focus
    (10, 1, 6),  # degenerate m_eff
    (10, 8, 4),
    (10, 4, 6),
]


@pytest.mark.parametrize("cell", CORRECTNESS_CELLS, ids=lambda c: f"K{c[0]}_m{c[1]}_hn{c[2]}")
@pytest.mark.parametrize(
    "variant",
    ("tree", "rs_flat_epi", "rs_m_epi", "rs_ragged_epi", "rs_ragged_unfused", "rs_flat_noepi", "rs_ragged_noepi"),
)
def test_correctness(device, variant, cell):
    k, m_eff, hn_pad = cell
    pcc, _, l1 = _run_cell(device, variant, k, m_eff, hn_pad, measure=False)
    logger.info(f"[correct] {variant:14s} K={k:2d} m_eff={m_eff} HN_PAD={hn_pad} pcc={pcc:.6f} l1={l1}")
    assert pcc > 0.99, f"{variant} K={k} m_eff={m_eff} HN_PAD={hn_pad}: pcc {pcc}"


# ---------------------------------------------------------------------------
# Device: the focus shape — the whole menu
# ---------------------------------------------------------------------------


def test_focus_menu(device):
    k, m_eff, hn_pad = FOCUS
    rows = []
    for variant in (
        "seed_only",
        "tree",
        "rs_flat_epi",
        "rs_m_epi",
        "rs_ragged_epi",
        "rs_flat_unfused",
        "rs_ragged_unfused",
        "rs_flat_noepi",
        "rs_ragged_noepi",
    ):
        pcc, ns, l1 = _run_cell(device, variant, k, m_eff, hn_pad)
        rows.append((variant, ns, pcc, l1))
        logger.info(f"[focus] {variant:14s} ns={ns:9.1f} pcc={pcc:.6f} l1={l1}")
    base = dict((v, n) for v, n, _, _ in rows)
    floor = base["seed_only"]
    logger.info(
        "\n=== reduce_scatter_swiglu FOCUS (K=%d m_eff=%d HN_PAD=%d, T=%d) ===\n" % (k, m_eff, hn_pad, m_eff * hn_pad)
        + "\n".join(f"{v:14s} ns={n:9.1f}  net_of_seed={n - floor:9.1f}  pcc={p:.6f}  l1={b}" for v, n, p, b in rows)
    )


# ---------------------------------------------------------------------------
# Device: the predicate sweep
# ---------------------------------------------------------------------------

SWEEP = [
    (10, 8, 6),  # focus
    (10, 4, 6),  # count 128 regime
    (10, 2, 6),
    (10, 1, 6),  # degenerate: 6 tiles over 10 cores
    (10, 8, 4),  # HN_PAD 4 (the op's ragged last column width)
    (8, 8, 6),
    (4, 8, 6),
    (2, 8, 6),  # degenerate KGROUPS
]


def test_full_grid_contention(device):
    """THE CAVEAT KILLER. Everything above measures ONE column with no competing traffic, which is
    exactly the objection that kept a round-1 collective idea from graduating. The candidate moves
    MORE NoC bytes than the tree (K senders x K destinations x 2 operands of a slice each, versus
    K-1 tree edges of a whole block), so the honest question is whether the win survives all HGROUPS
    columns running their collective AT THE SAME TIME. This runs the focus geometry on the op's real
    HGROUPS x KGROUPS = 11 x 10 = 110-core grid, all 11 columns concurrent, and gates PCC on ALL 11
    roots (the min over columns)."""
    k, m_eff, hn_pad = FOCUS
    rows = []
    for variant in ("seed_only", "tree", "rs_flat_epi", "rs_ragged_epi", "rs_ragged_unfused", "rs_flat_noepi"):
        pcc, ns, l1 = _run_cell(device, variant, k, m_eff, hn_pad, ncols=11)
        rows.append((variant, ns, pcc, l1))
        logger.info(f"[grid11] {variant:18s} ns={ns:9.1f} min_pcc_over_11_roots={pcc:.6f} l1={l1}")
        if variant != "seed_only":
            assert pcc > 0.99, f"{variant} on the 110-core grid: min PCC {pcc}"
    floor = dict((v, n) for v, n, _, _ in rows)["seed_only"]
    logger.info(
        "\n=== reduce_scatter_swiglu FULL GRID (11 columns x 10 rows = 110 cores, focus geometry) ===\n"
        + "\n".join(f"{v:18s} ns={n:9.1f}  net_of_seed={n - floor:9.1f}  pcc={p:.6f}  l1={b}" for v, n, p, b in rows)
    )


def test_predicate_sweep(device):
    variants = ("tree", "rs_flat_epi", "rs_m_epi", "rs_ragged_epi")
    rows = []
    for k, m_eff, hn_pad in SWEEP:
        for variant in variants:
            pcc, ns, l1 = _run_cell(device, variant, k, m_eff, hn_pad)
            rows.append((k, m_eff, hn_pad, variant, ns, pcc, l1))
            logger.info(f"[sweep] K={k:2d} m_eff={m_eff:2d} HN={hn_pad} {variant:12s} ns={ns:9.1f} pcc={pcc:.6f}")
    logger.info(
        "\n=== reduce_scatter_swiglu SWEEP ===\n"
        + "\n".join(
            f"K={k:2d} m_eff={m:2d} HN={h} {v:12s} ns={n:9.1f} pcc={p:.6f} l1={b}" for k, m, h, v, n, p, b in rows
        )
    )
