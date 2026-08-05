# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: can rms_norm's pass B be ONE chain instead of two?

CORRECTNESS is the only pass/fail (fp64 torch reference, pcc >= 0.9995 and
rel-RMS <= 0.04 -- the focus case's soft gates).  Perf is MEASURED, never asserted.

Run:
    scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/pass_b_fuse_scale_gamma/test_pass_b_fuse_scale_gamma.py \
        -k test_focus
then read DEVICE KERNEL DURATION [ns] out of the printed
generated/profiler/reports/<ts>/ops_perf_results_*.csv.  EVERY test performs exactly
ONE `ttnn.generic_op`, so CSV row k is manifest line k; the manifest (execution order
+ pcc + rel-RMS per row) is written to `last_run_manifest.txt` in this directory at
session teardown, and `report.py` joins the two.

Env selectors (comma lists) to narrow a re-run:
    PBF_VARIANTS=baseline,fused_blk      PBF_POINTS=8x4,1x32
"""

import importlib.util
import os
from pathlib import Path

import pytest
import ttnn

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("pbf_bench", _HERE / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

TILE = bench.TILE

# The focus case's soft gates, on the op's NORMALIZED OUTPUT -- which is exactly what
# this bench produces, so they apply directly (no reinterpretation as in the pass-A
# benches).
PCC_GATE = 0.9995
RELRMS_GATE = 0.04

EPS = 1e-12

# Focus geometry: one core of (1,1,8192,1024) BLOCK_SHARDED shard [1024,128] on an
# (8,8) grid -- 32 tile-rows x Wt=4, BLOCK_ROWS=8 -> 4 blocks of 8x4 = 32 tiles.
FOCUS = dict(rows=8, wt=4, blocks=4)


def _blocks_for(rows, wt):
    """Row-blocks such that the core walks <= 32 tile-rows and <= 128 tiles.

    128 tiles == the focus case's per-core pass-B payload, and 32 tile-rows == its
    per-core assignment; both caps keep the fp32 stat CB and the bf16 x/out shards
    inside one core's L1 at every sweep point.
    """
    return max(1, min(32 // rows, 128 // (rows * wt)))


SWEEP_POINTS = [(r, w) for r in (1, 8, 32) for w in (1, 4, 16, 32) if r * w <= 128]


def _sel(env, allowed):
    v = os.environ.get(env)
    if not v:
        return list(allowed)
    want = set(v.split(","))
    bad = want - {str(a) for a in allowed}
    if bad:
        raise ValueError(f"unknown {env}: {sorted(bad)}; valid: {list(allowed)}")
    return [a for a in allowed if str(a) in want]


ALL_VARIANTS = _sel("PBF_VARIANTS", list(bench.VARIANTS))
POINT_IDS = {f"{r}x{w}": (r, w) for r, w in SWEEP_POINTS}
SWEEP = [POINT_IDS[k] for k in _sel("PBF_POINTS", list(POINT_IDS))]

# The sweep runs a NARROW variant set (one device run each) unless overridden.
SWEEP_VARIANTS = _sel("PBF_SWEEP_VARIANTS", ["baseline", "baseline_blk", "baseline_blk_up", "fused_blk"])


# ---------------------------------------------------------------------------
# Manifest: execution-ordered join key for the profiler CSV.
# ---------------------------------------------------------------------------
_MANIFEST = []


@pytest.fixture(scope="module", autouse=True)
def _write_manifest():
    yield
    path = _HERE / "last_run_manifest.txt"
    with open(path, "w") as f:
        f.write("# row  variant  rows  wt  blocks  tiles  gamma  blk  pcc  rel_rms\n")
        for i, m in enumerate(_MANIFEST, start=1):
            f.write(
                f"{i}\t{m['variant']}\t{m['rows']}\t{m['wt']}\t{m['blocks']}\t"
                f"{m['tiles']}\t{int(m['gamma'])}\t{m['blk']}\t{m['pcc']:.6f}\t{m['rel_rms']:.5f}\n"
            )
    print(f"\n[pass_b_fuse_scale_gamma] manifest ({len(_MANIFEST)} device runs): {path}")


# ---------------------------------------------------------------------------
# Inputs + fp64 reference
# ---------------------------------------------------------------------------
def _make_tensors(device, rows, wt, blocks, has_gamma):
    import torch

    torch.manual_seed(20260805)
    rows_total = rows * blocks
    h, w = rows_total * TILE, wt * TILE

    x = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    # 1/rms, per LOGICAL row, from the bf16 x -- i.e. the value the op's pass A +
    # finalize would deliver for this x.
    inv_rms = torch.rsqrt(x.to(torch.float64).pow(2).mean(dim=-1) + EPS).to(torch.float32)
    # POISONED stat tile: column 0 is the real value, every other column is 1e3x
    # wrong.  A variant that reads any lane but column 0 fails the pcc gate instead
    # of passing by luck (the same guard root_finalize_scope uses).
    stat = (inv_rms[:, None] * 1.0e3).repeat(1, TILE)
    stat[:, 0] = inv_rms

    gamma_row = (torch.rand((w,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    gamma = torch.zeros((TILE, w), dtype=torch.bfloat16)
    gamma[0, :] = gamma_row  # row-shaped: valid in row 0 only, as the op stages it

    x_t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, w)),
    )
    stat_t = ttnn.from_torch(
        stat,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, TILE)),
    )
    gamma_t = (
        ttnn.from_torch(
            gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bench.sharded_memory_config((TILE, w)),
        )
        if has_gamma
        else None
    )
    out_t = ttnn.from_torch(
        torch.zeros((h, w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, w)),
    )

    ref = x.to(torch.float64) * inv_rms.to(torch.float64)[:, None]
    if has_gamma:
        ref = ref * gamma_row.to(torch.float64)[None, :]
    return x_t, stat_t, gamma_t, out_t, ref


def _pcc_relrms(got, ref):
    import torch

    got, ref = got.to(torch.float64), ref.to(torch.float64)
    gc, rc = got - got.mean(), ref - ref.mean()
    denom = (gc.norm() * rc.norm()).item()
    pcc = 1.0 if denom == 0 else (gc * rc).sum().item() / denom
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    return pcc, rel_rms


def _run(device, variant, rows, wt, blocks, has_gamma=True, blk=None):
    x_t, stat_t, gamma_t, out_t, ref = _make_tensors(device, rows, wt, blocks, has_gamma)
    out = bench.run(
        x_t, stat_t, gamma_t, out_t, variant=variant, rows=rows, wt=wt, blocks=blocks, has_gamma=has_gamma, blk=blk
    )
    got = ttnn.to_torch(out).to(ref.dtype)
    pcc, rel_rms = _pcc_relrms(got, ref)
    tiles = rows * wt * blocks
    eff_blk = bench.blk_for(wt) if blk is None else blk
    _MANIFEST.append(
        dict(
            variant=variant,
            rows=rows,
            wt=wt,
            blocks=blocks,
            tiles=tiles,
            gamma=has_gamma,
            blk=eff_blk,
            pcc=pcc,
            rel_rms=rel_rms,
        )
    )
    print(
        f"\n[pass_b_fuse_scale_gamma] row={len(_MANIFEST)} variant={variant} rows={rows} wt={wt} "
        f"blocks={blocks} tiles={tiles} gamma={int(has_gamma)} blk={eff_blk}  "
        f"pcc={pcc:.6f} rel_rms={rel_rms:.5f}"
    )
    assert pcc >= PCC_GATE, f"{variant} rows={rows} wt={wt}: pcc {pcc} < {PCC_GATE}"
    assert rel_rms <= RELRMS_GATE, f"{variant} rows={rows} wt={wt}: rel_rms {rel_rms} > {RELRMS_GATE}"
    return got


# ---------------------------------------------------------------------------
# 1. The focus geometry, every variant, in report order.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_focus(device, variant):
    _run(device, variant, FOCUS["rows"], FOCUS["wt"], FOCUS["blocks"])


# ---------------------------------------------------------------------------
# 2. Domain sweep: rows-per-block x width tiles per chunk.
#    rows=1 + wide  == the WIDTH-shard decode profile
#    rows=32        == the prefill profile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows, wt", SWEEP, ids=[f"{r}x{w}" for r, w in SWEEP])
@pytest.mark.parametrize("variant", SWEEP_VARIANTS)
def test_sweep(device, variant, rows, wt):
    _run(device, variant, rows, wt, _blocks_for(rows, wt))


# ---------------------------------------------------------------------------
# 3. HAS_GAMMA == 0: the op's no-gamma mode.  There is nothing to fuse, so the
#    fused variant is the SAME code as the baseline -- measured to prove it, and to
#    give the scale-only cost that decomposes the baseline's two passes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", [v for v in ("baseline", "baseline_blk", "fused_blk") if v in ALL_VARIANTS])
def test_no_gamma(device, variant):
    _run(device, variant, FOCUS["rows"], FOCUS["wt"], FOCUS["blocks"], has_gamma=False)


# ---------------------------------------------------------------------------
# 3b. The DEST-lane amortization CURVE.  Same 128-tile payload, same chain
#     structure, only `blk` moves -- so this is the shape of the per-outer-iter
#     fixed cost (tile_regs handshake + per-element init/reconfig + CB ops).
#     wt=8 so blk can reach the DEST ceiling of 8 bf16 tiles.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("blk", [1, 2, 4, 8])
@pytest.mark.parametrize("variant", [v for v in ("baseline_blk", "fused_blk") if v in ALL_VARIANTS])
def test_blk_curve(device, variant, blk):
    _run(device, variant, rows=8, wt=8, blocks=2, blk=blk)


# ---------------------------------------------------------------------------
# 3c. DOMAIN follow-ups (all selectable with `-k test_dom`).
#
#  rows_curve  the crossover axis.  128 tiles and blk=4 fixed, only `rows` (==
#              BLOCK_ROWS) moves, so the CALL COUNT moves: the baseline runs
#              2*blocks chain calls and the fused variant runs blocks.  This is
#              what decides which spelling wins.
#  decode      the REAL width-shard decode geometry: ONE tile-row per core, one
#              row-block -- i.e. 2 chain calls total (baseline) vs 1 (fused).
#  fused_wide  the sweep cells where blk_for(wt) picked 8 and the fused variant is
#              NUMERICALLY WRONG there (pcc 0.987); re-measured at the largest blk
#              where it is correct.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("variant", [v for v in ("baseline_blk", "fused_blk") if v in ALL_VARIANTS])
def test_dom_rows_curve(device, variant, rows):
    _run(device, variant, rows, 4, 32 // rows, blk=4)


@pytest.mark.parametrize("rows, wt, blocks", [(1, 4, 1), (1, 16, 1), (1, 32, 1)], ids=["1x4x1", "1x16x1", "1x32x1"])
@pytest.mark.parametrize("variant", [v for v in ("baseline", "baseline_blk", "fused_blk") if v in ALL_VARIANTS])
def test_dom_decode(device, variant, rows, wt, blocks):
    _run(device, variant, rows, wt, blocks, blk=min(4, wt))


@pytest.mark.parametrize("rows, wt, blocks", [(1, 16, 8), (1, 32, 4), (8, 16, 1)], ids=["1x16", "1x32", "8x16"])
def test_dom_fused_wide(device, rows, wt, blocks):
    _run(device, "fused_blk", rows, wt, blocks, blk=4)


# ---------------------------------------------------------------------------
# 4. Does the fusion change the VALUES?  At fp32_dest_acc_en=False a DEST word is
#    16-bit, so `x*stat` is rounded to bf16 in DEST either way -- packing it to a
#    bf16 CB and unpacking it back should be bit-for-bit the same as reading it
#    straight out of DEST.  Reported, and gated only relative to the baseline.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", [v for v in bench.VARIANTS if v != "baseline"])
def test_matches_baseline(device, variant):
    import torch

    b = _run(device, bench.BASELINE, FOCUS["rows"], FOCUS["wt"], FOCUS["blocks"])
    c = _run(device, variant, FOCUS["rows"], FOCUS["wt"], FOCUS["blocks"])
    identical = torch.equal(b, c)
    max_abs = (b - c).abs().max().item()
    print(f"[pass_b_fuse_scale_gamma] {variant} vs baseline: bitwise={identical} max_abs_diff={max_abs:.3e}")
