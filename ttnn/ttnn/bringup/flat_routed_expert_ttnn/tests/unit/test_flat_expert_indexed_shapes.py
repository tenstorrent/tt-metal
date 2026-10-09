# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""flat_routed_expert indexed mode (token_index: flat row r reads gathered row token_index[r]), x_pages_per_row and
y_row_major across the plan families of test_flat_routed_expert_op.py (MiMo 64 experts, K2 reader tails, TP4 2
subgrids, TP2 3 subgrids), one chip:

* test_flat_expert_indexed_shapes: indexed y bit-identical to the flat dispatch buffer's y, for x pages per row 1,
  H / 1024 and H / 512; y_row_major within bfp8 rounding of the tiled y; two launches with different counts and a
  moved arena (the cache-hit address patch).
* test_flat_expert_indexed_determinism: repeated indexed launches bitwise identical on the active rows (compared on
  device; rows outside the active regions are never written, so they are masked out).
* test_flat_expert_indexed_perf: real-time profiler device time per (case, mode) against baselines.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from ttnn.bringup.flat_routed_expert_ttnn.tests.unit.test_flat_routed_expert_op import CASES, _counts
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler


def _op(device, case):
    tag, H, I, E, m, NG = case
    torch.manual_seed(0)
    weights = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    gids = [[(4 * e + 1) % NG if E * 4 <= NG else (2 * e + 1) % NG for e in range(E)]]
    return FlatRoutedExpert(device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, pin=1), gids[0]


def _indexed_inputs(device, case, gids, counts, seed):
    """Gathered x [m, H] (m = the capacity: every expert can take every gathered token), counts / regions, token_index
    [1, rows] (each active expert's rows: distinct ascending gathered tokens), the equivalent flat buffer, and the
    active-row mask [rows, 1]."""
    tag, H, I, E, m, NG = case
    g = torch.Generator().manual_seed(seed)
    offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32 + 64
    x = torch.randn(m, H, generator=g)
    tok = torch.zeros(rows, dtype=torch.int32)
    mask = torch.zeros(rows, 1)
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(gids):
        t = torch.randperm(m, generator=g)[: counts[e]].sort().values
        tok[offs[e] : offs[e] + counts[e]] = t.int()
        mask[offs[e] : offs[e] + counts[e]] = 1
        c_[0, gid], r_[0, gid] = counts[e], offs[e]
    flat = torch.zeros(rows, H)
    flat[mask[:, 0] > 0] = x[tok[mask[:, 0] > 0].long()]
    rm = lambda t, dt: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    d = dict(
        x={P: rm(x.reshape(m * P, H // P), ttnn.bfloat16) for P in sorted({1, H // 1024, H // 512})},
        flat=rm(flat, ttnn.bfloat16),
        counts=rm(c_, ttnn.uint32),
        regions=rm(r_, ttnn.uint32),
        tok=rm(tok[None], ttnn.uint32),
        mask=ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )
    return d, offs, mask


def _modes(H):
    """mode -> (x pages per row or None for the flat buffer, y_row_major)"""
    m = {"flat": (None, False), "indexed": (1, False), "indexed_rm": (1, True)}
    m[f"indexed_p{H // 1024}"] = (H // 1024, False)
    m[f"indexed_p{H // 512}"] = (H // 512, False)
    return m


def _call(op, d, P, y_rm):
    if P is None:
        return op(d["flat"], d["counts"], d["regions"], y_row_major=y_rm)
    return op(d["x"][P], d["counts"], d["regions"], token_index=d["tok"], x_pages_per_row=P, y_row_major=y_rm)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_flat_expert_indexed_shapes(device, case):
    tag, H, I, E, m, NG = case
    op, gids = _op(device, case)
    pinned = []
    for launch, seed in enumerate((1, 2)):
        counts = _counts(E, m, seed)
        d, offs, mask = _indexed_inputs(device, case, gids, counts, seed)
        act = mask[:, 0] > 0
        ys = {}
        for mode, (P, y_rm) in _modes(H).items():
            ys[mode] = ttnn.to_torch(_call(op, d, P, y_rm)).float()[act]
        for mode in ys:
            if mode in ("flat", "indexed_rm"):
                continue
            assert torch.equal(ys[mode], ys["flat"]), f"{tag} launch {launch}: {mode} y differs from the flat buffer's"
        # bf16 row-major vs bfp8 tiles: bfp8 has 7 mantissa bits with an exponent shared by 16 values
        b = ys["flat"]
        blk = b.abs().reshape(b.shape[0], -1, 16).amax(-1, keepdim=True).expand(-1, -1, 16).reshape(b.shape)
        worst = ((ys["indexed_rm"] - b).abs() / (blk + 1e-30)).max().item()
        assert worst <= 2**-6, f"{tag} launch {launch}: row-major y off by {worst} of the block max"
        logger.info(
            f"{tag} launch {launch}: {int(act.sum())} active rows; indexed (P 1, {H // 1024}, {H // 512}) bit-identical "
            f"to flat; row-major within {worst:.2e}"
        )
        pinned.append(  # the next launch's arena sits lower: the cache hit must patch every address
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )


def _mismatch_marker(reference, actual, mask):
    """On-device exact compare of the active rows (as tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py):
    non-zero iff any active element differs; outputs never leave the device."""
    to_tile = lambda t: t if t.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    ne = ttnn.ne(to_tile(reference), to_tile(actual), dtype=ttnn.bfloat16)
    return ttnn.max(ttnn.multiply(ne, mask))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("y_rm", [False, True], ids=["y_bfp8", "y_rm"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_flat_expert_indexed_determinism(device, case, y_rm):
    tag, H, I, E, m, NG = case
    op, gids = _op(device, case)
    d, _, _ = _indexed_inputs(device, case, gids, _counts(E, m, 1), 1)
    P = H // 1024
    reference = _call(op, d, P, y_rm)
    marker = None
    for _ in range(19):
        y = _call(op, d, P, y_rm)
        mk = _mismatch_marker(reference, y, d["mask"])
        marker = mk if marker is None else ttnn.maximum(marker, mk)
        y.deallocate(True)
    assert float(ttnn.to_torch(marker).item()) == 0.0, f"{tag}: flat_routed_expert (indexed) is not deterministic"


# Device ns per (case, mode), median of 3, BH p150 (QuietBox chip), counts _counts(E, m, 1). Recalibrate from the
# "RT-CAL" lines (2026-09-29). Indexed costs nothing over the flat buffer; 1 KB x pages (P = H / 512) cost K2 / TP4
# +32-38% (1 KB NoC reads), 2 KB pages (P = H / 1024) are free.
_PERF_EXPECTED_NS = {
    ("mimo", "flat"): 1_933_323,
    ("mimo", "indexed"): 1_942_624,
    ("mimo", "indexed_rm"): 1_971_038,
    ("mimo", "indexed_p4"): 1_972_610,
    ("mimo", "indexed_p8"): 1_944_741,
    ("k2", "flat"): 1_581_504,
    ("k2", "indexed"): 1_581_722,
    ("k2", "indexed_rm"): 1_599_456,
    ("k2", "indexed_p7"): 1_580_836,
    ("k2", "indexed_p14"): 2_086_615,
    ("tp4", "flat"): 923_968,
    ("tp4", "indexed"): 927_059,
    ("tp4", "indexed_rm"): 978_608,
    ("tp4", "indexed_p7"): 926_166,
    ("tp4", "indexed_p14"): 1_275_363,
    ("tp2", "flat"): 1_390_231,
    ("tp2", "indexed"): 1_392_361,
    ("tp2", "indexed_rm"): 1_398_004,
    ("tp2", "indexed_p7"): 1_392_007,
    ("tp2", "indexed_p14"): 1_390_647,
}
_PERF_MARGIN = 0.03


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_flat_expert_indexed_perf(device, case):
    require_realtime_profiler("flat_routed_expert perf checks")
    tag, H, I, E, m, NG = case
    op, gids = _op(device, case)
    d, _, _ = _indexed_inputs(device, case, gids, _counts(E, m, 1), 1)
    missing = []
    for mode, (P, y_rm) in _modes(H).items():
        expected = _PERF_EXPECTED_NS.get((tag, mode))
        assert_op_duration_merged(
            device,
            lambda: _call(op, d, P, y_rm).deallocate(True),
            "/flat_routed_expert_ttnn/",
            expected_ns=expected or 1,
            margin=_PERF_MARGIN if expected else float("inf"),
            label=f'("{tag}", "{mode}")',
            iters=3,
        )
        if expected is None:
            missing.append(mode)
    if missing:
        pytest.skip(f"no baseline for {tag} {missing}; add them to _PERF_EXPECTED_NS")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_expert_x_page_straddle_rejected(device, expect_error):
    """x pages that are neither a multiple nor a divisor of the relay's 2 KB read segment would make a segment read
    run past its page into the wrong bank: H 7168 with 2 pages per row (7 KB pages) must be rejected."""
    case = CASES[1]  # k2, H 7168
    tag, H, I, E, m, NG = case
    op, gids = _op(device, case)
    counts = _counts(E, m, 1)
    d, _, _ = _indexed_inputs(device, case, gids, counts, 1)
    x2 = ttnn.from_torch(
        torch.randn(m * 2, H // 2),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "read segment"):
        op(x2, d["counts"], d["regions"], token_index=d["tok"], x_pages_per_row=2)
