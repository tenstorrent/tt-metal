# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: which reduce DATAPATH belongs at which PER-CALL reduce width?

CORRECTNESS is the only pass/fail (vs an fp64 torch reference).  Perf is MEASURED,
never asserted: run under `scripts/run_safe_pytest.sh --profile ...` and read
DEVICE KERNEL DURATION [ns] out of
generated/profiler/reports/*/ops_perf_results_*.csv, one row per test in order.

Tests, in the order the profiler CSV reports them:

  test_focus_menu[...]    the FOCUS geometry (rms_norm (1,1,8192,1024) BLOCK_SHARDED
                          shard [1024,128] -> per core 32 tile-rows, BLOCK_ROWS=8,
                          WT_CHUNK=4, X_SQUARED_WT=1), pass A = square + reduce.
                          The full option menu: the op's datapath today, the predicate
                          fix, and the two fused spellings.
  test_crossover[...]     THE crossover measurement: per-call reduce width x tile-rows
                          per call, BOTH datapaths, reduce alone.
  test_unfolded[...]      the same question in the op's pass-A geometry with the D12
                          fold OFF (X_SQUARED_WT == WT_CHUNK), which is where the
                          existing REDUCE_ACC_VIA_ADD knob was measured.
  test_partial_w[...]     the COUPLED partial-W mechanism (partial scaler pair vs 0/1
                          mask), pad lanes POISONED so a leak is catastrophic.

GATE CHOICE, stated explicitly.  This bench's output is the RAW sum(x^2) -- the
quantity just before rms_norm's rsqrt, which compresses relative error ~2x and then
divides it out.  A raw bf16 sum of 32*width all-positive addends therefore has a
legitimately lower PCC than the op's normalized output the 0.9995 soft gate is
declared on.  So the HARD gate here is rel-RMS <= 0.04 (the op's other soft gate,
dimensionally meaningful on a raw sum), plus a POISON gate on the partial-W points
(a padding leak blows rel-RMS past any threshold).  PCC is REPORTED at every point,
because a per-option precision number is exactly what the coordinator's menu needs.
"""

import importlib.util
from pathlib import Path

import pytest
import ttnn

_spec = importlib.util.spec_from_file_location("rapw1_bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

RELRMS_GATE = 0.04
PCC_REPORT_FLOOR = 0.9995  # reported, not enforced -- see the header

# The focus geometry, per the coordinator's brief.
FOCUS = dict(rows=8, width=4, blocks=4)

# L1 cap: cb_in is rows*width bf16 tiles (2 kB each) + cb_out is rows fp32 tiles (4 kB)
# + (mode 1) cb_x_squared.  256 bf16 tiles = 512 kB keeps every point inside one core's
# ~1.4 MB budget with room for the fp32 stat CB -- the op is bounded the same way.
MAX_IN_TILES = 256

CROSSOVER = [
    pytest.param(rows, width, id=f"rows{rows}_w{width}")
    for rows in (1, 8, 32)
    for width in (1, 2, 4, 8, 16, 32)
    if rows * width <= MAX_IN_TILES
]


def _blocks_for(rows):
    """Keep the TOTAL tile-rows reduced at ~32 (the focus shape's per-core count)."""
    return max(1, 32 // rows)


def _pcc_relrms(got, ref):
    import torch

    got = got.to(torch.float64)
    ref = ref.to(torch.float64)
    gc, rc = got - got.mean(), ref - ref.mean()
    denom = (gc.norm() * rc.norm()).item()
    pcc = 1.0 if denom == 0 else (gc * rc).sum().item() / denom
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    return pcc, rel_rms


def _make_tensors(device, *, rows, width, pre_squared, partial_w=0, poison=0.0):
    """The bf16 reduce/square input and the fp32 stat output, both L1-sharded on core (0,0).

    pre_squared  MODE 0 feeds cb_in straight to the reduce, so it must hold x^2.
                 MODE 1/2 square it in the kernel, so it holds x.
    partial_w    valid reduce-dim elements in the LAST width tile.  The pad lanes
                 [partial_w, 32) are filled with `poison` so a leak is catastrophic.
    """
    import torch

    torch.manual_seed(1234)
    h, w = rows * bench.TILE, width * bench.TILE
    x = torch.randn((h, w), dtype=torch.float32)
    if partial_w:
        valid = (width - 1) * bench.TILE + partial_w
        if poison:
            x[:, valid:] = poison
        real = x[:, :valid]
    else:
        real = x

    if pre_squared:
        payload = (x * x).to(torch.bfloat16)
        ref = (real * real).to(torch.bfloat16).to(torch.float64).sum(dim=-1)
        if partial_w and poison:
            # keep the poison exactly as the host wrote it (x^2 of a poison lane)
            payload[:, valid:] = (poison * poison) if abs(poison) < 1e18 else poison
    else:
        payload = x.to(torch.bfloat16)
        ref = payload[:, : real.shape[-1]].to(torch.float64).pow(2).sum(dim=-1)

    x_in = ttnn.from_torch(
        payload,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, w)),
    )
    stat_out = ttnn.from_torch(
        torch.zeros((h, bench.TILE), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, bench.TILE)),
    )
    return x_in, stat_out, ref


def _run(device, variant, *, rows, width, blocks, fold=True, partial_w=0, poison=0.0, label=""):
    mode = bench.VARIANT_SPEC[variant][0]
    x_in, stat_out, ref = _make_tensors(
        device,
        rows=rows,
        width=width,
        pre_squared=(mode == bench.MODE_REDUCE_ONLY),
        partial_w=partial_w,
        poison=poison,
    )
    out = bench.run(
        x_in, stat_out, variant=variant, rows=rows, width=width, blocks=blocks, partial_w=partial_w, fold=fold
    )
    got = ttnn.to_torch(out)[:, 0]
    pcc, rel_rms = _pcc_relrms(got, ref)
    per_call_w = width if (mode == bench.MODE_REDUCE_ONLY or not fold) else 1
    flag = "" if pcc >= PCC_REPORT_FLOOR else "  (below 0.9995 -- raw-sum PCC, see header)"
    print(
        f"\n[reduce_at_percall_width_1]{label} variant={variant} rows={rows} wt_chunk={width} "
        f"per_call_w={per_call_w} blocks={blocks} partial_w={partial_w}  "
        f"pcc={pcc:.6f} rel_rms={rel_rms:.5f}{flag}"
    )
    assert rel_rms <= RELRMS_GATE, f"{variant} rows={rows} width={width} partial={partial_w}: rel_rms {rel_rms}"
    return got, pcc, rel_rms


# ---------------------------------------------------------------------------
# 1. THE FOCUS GEOMETRY -- the full option menu, in one place.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", bench.PASS_A)
def test_focus_menu(device, variant):
    _run(device, variant, label=" FOCUS", **FOCUS)


# ---------------------------------------------------------------------------
# 2. THE CROSSOVER -- both datapaths at every (per-call width, rows) point.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows, width", CROSSOVER)
@pytest.mark.parametrize("variant", bench.DATAPATHS)
def test_crossover(device, variant, rows, width):
    _run(device, variant, rows=rows, width=width, blocks=_blocks_for(rows), label=" XOVER")


# ---------------------------------------------------------------------------
# 3. The D12 fold OFF (X_SQUARED_WT == WT_CHUNK) in the op's pass-A geometry --
#    the regime the existing REDUCE_ACC_VIA_ADD knob was measured on.  A predicate
#    that wins at per-call width 1 by losing here is not a win.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [4, 16])
@pytest.mark.parametrize("variant", ["sq_acc_add", "sq_reduce_tile"])
def test_unfolded(device, variant, width):
    _run(device, variant, rows=8, width=width, blocks=4, fold=False, label=" UNFOLD")


# ---------------------------------------------------------------------------
# 3b. THE DOMAIN of the predicate change, in pass-A geometry.  The predicate flips
#     exactly where the D12 fold is ON, i.e. tile-aligned W with
#     2 <= WT_CHUNK <= DEST_ACC_SQUARE_MAX_WT(8).  Sweep that whole WT_CHUNK range
#     x the row-block sizes the op actually uses (decode 1, focus 8, full 32).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("variant", ["sq_acc_add", "sq_reduce_tile"])
def test_domain_passA(device, variant, rows, width):
    _run(device, variant, rows=rows, width=width, blocks=_blocks_for(rows), label=" DOMAIN")


# ---------------------------------------------------------------------------
# 4. The COUPLED partial-W mechanism, pad lanes POISONED.
#    ReduceTile routes a PARTIAL SCALER tile to the last width tile (SCALER_TILES 2);
#    AccumulateViaAdd folds it with a 0/1 MASK tile (SCALER_TILES 1).  Both must zero
#    the pad lanes exactly.  P == 8 is what every one of feature_spec's
#    _PAD_POISON_SHAPES (W in {40,72,136,200}) reduces to; 1 and 31 bracket it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width, partial_w", [(2, 8), (5, 8), (2, 1), (2, 31)])
@pytest.mark.parametrize("variant", bench.DATAPATHS)
def test_partial_w(device, variant, width, partial_w):
    _run(device, variant, rows=32, width=width, blocks=1, partial_w=partial_w, poison=1.0e4, label=" PARTIAL")
