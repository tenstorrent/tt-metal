# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off runner for `gather_slot_major_coalesce`.

    scripts/run_safe_pytest.sh --profile --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/gather_slot_major_coalesce/test_gather_slot_major_coalesce.py \
        -k focus
    python3 .../gather_slot_major_coalesce/read_results.py focus

Correctness is the ONLY pass/fail; perf is recorded, never asserted.  The gate is

  * BIT-EXACTNESS against the baseline (`rm_f2`) at the same BLOCK_ROWS.  The layout
    permutes which page a partial lands on; the fold's operand PAIRS and their ORDER are
    unchanged, and every faces value ships face 0 and face 2 (where the only lanes the
    consumer reads live).  So anything other than an identical output is a bug in this
    bench, not a precision trade.
  * pcc AND rel-RMS against a torch fp32 reference.  rel-RMS is not optional: this op has
    caught two bugs that held pcc >= 0.9997 and showed ONLY in rel-RMS (a ~1000x uniform
    scale error from a missing reconfig_data_format).

`rm_f2_nozero` is an ABLATION, not a proposal -- the baseline with the `writer_gather_zero`
PAYLOAD stripped and every CB handshake / trip count intact.  It exists to price the boot
independently of the coalescing.  It is expected to be correct-by-accident here (the faces
it leaves undefined are never read back) but is recorded, not gated, because "undefined L1
happens to be harmless" is not a correctness argument.

Every variant of a case runs in ONE profiled process, one program per variant, and the test
writes `manifests/<case>.jsonl` (one line per launch, in launch order) so read_results.py
can join positionally to the profiler CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd

from . import bench


class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


HERE = Path(__file__).parent
MANIFEST_DIR = HERE / "manifests"

_ML = ttnn.TensorMemoryLayout

# (shape, [shard_shape, grid], memory_layout, block_rows_cap, grid_w)
#
# `block_rows_cap` is the only synthetic knob: it lowers the descriptor's own
# L1_SAFETY_FRACTION until its own solve picks BLOCK_ROWS <= cap, which is how the
# BLOCK_ROWS / num_blocks axis is swept without changing the shard geometry.
# None = the op's real solve.
CASES = {
    # ---- THE FOCUS GEOMETRY: 64 cores, 8 groups of 8, BLOCK_ROWS 8, 4 rounds --------
    "focus": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, None, None),
    # ---- BLOCK_ROWS axis at GROUP_SIZE 8 (the task's {1, 8, 32}) --------------------
    # BLOCK_ROWS == 1 has NOTHING to coalesce (one tile-row per round) and is the
    # expected-flat control.  BLOCK_ROWS == 32 is attempted and is expected to be
    # L1-UNREACHABLE at GROUP_SIZE 8 (a 32-row gather ring is 8*32*4 kB = 1 MB/core);
    # that is a property of the OP's L1 solve, identical for baseline and candidate, and
    # is recorded as such rather than asserted away.
    "br1": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 1, None),
    "br4": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 4, None),
    "br32": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 32, None),
    # ---- GROUP_SIZE axis {4, 8, 9, 28, 32} ------------------------------------------
    "gs4": ((1, 1, 8192, 1024), ([1024, 256], (4, 8)), _ML.BLOCK_SHARDED, None, None),
    "gs4_br1": ((1, 1, 8192, 1024), ([1024, 256], (4, 8)), _ML.BLOCK_SHARDED, 1, None),
    # ODD GROUP_SIZE -- the op's only odd live profile.  The pad slot must stay DEFINED,
    # so the boot survives here in BOTH layouts; slot-major makes the pad a single
    # CONTIGUOUS tail run instead of a strided scatter.
    "gs9": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, None, None),
    "gs28": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, None, None),
    "gs32": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, None, None),
    "gs32_multi": ((1, 1, 2048, 1024), ([2048, 32], (8, 4)), _ML.WIDTH_SHARDED, None, None),
    "gs16_multi": ((1, 1, 4096, 1024), ([4096, 64], (8, 2)), _ML.WIDTH_SHARDED, None, None),
    # ---- the NON-NATIVE input path: an INTERLEAVED width split (reader-fed input CB) --
    "ilv_gw8": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, None, 8),
}

FOCUS_CASES = ["focus"]
SWEEP_A = ["br1", "br4", "br32", "gs4", "gs4_br1"]
SWEEP_B = ["gs9", "gs28", "gs32", "gs32_multi", "gs16_multi", "ilv_gw8"]

PCC_GATE = 0.9995  # the focus case's soft pcc_threshold
RELRMS_GATE = 0.04  # the op's regression-net rms bound


def _mk_tensors(device, shape, shard, layout):
    from eval.sharding import shard_config

    torch.manual_seed(42)
    W = shape[-1]
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if shard is None
        else shard_config(shard[0], shard[1], layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
    x_t = torch.randn(shape, dtype=torch.bfloat16)
    g_t = torch.randn(W, dtype=torch.bfloat16)
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    g = ttnn.from_torch(g_t.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x_t, g_t, x, g, mc


def _resolve_safety_fraction(x, out, g, cfg, cap):
    """Lowest-perturbation L1_SAFETY_FRACTION whose own solve gives BLOCK_ROWS <= cap.

    Returns None when no fraction reaches the cap -- which for a cap ABOVE the op's own
    solve (br32) is the honest answer "this BLOCK_ROWS is not reachable on this geometry",
    a property of the op's L1 solve and identical for every variant.
    """
    if cap is None:
        return rpd.L1_SAFETY_FRACTION
    orig = rpd.L1_SAFETY_FRACTION
    try:
        # First: does the op's own solve already satisfy the cap?
        pd = rpd.create_program_descriptor(x, out, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
        if pd.kernels[2].compile_time_args[3] <= cap:
            return orig
        f = orig
        while f > 0.02:
            rpd.L1_SAFETY_FRACTION = f
            pd = rpd.create_program_descriptor(x, out, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
            if pd.kernels[2].compile_time_args[3] <= cap:
                return f
            f = round(f - 0.01, 4)
        return None
    finally:
        rpd.L1_SAFETY_FRACTION = orig


def _body(device, case, menu, repeat=1):
    shape, shard, layout, br_cap, grid_w = CASES[case]
    eps = 1e-6
    cfg = bench._perf_config()

    grid_w_orig = rpd.GRID_W
    rpd.GRID_W = grid_w if grid_w is not None else rpd.GRID_W
    try:
        x_t, g_t, x, g, mc = _mk_tensors(device, shape, shard, layout)
        ref = bench.torch_reference(x_t, g_t, eps)
        # ONE output tensor for every variant: a second resident shard per variant would eat
        # the L1 the CBs need (shards and the CB arena share L1).
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        frac = _resolve_safety_fraction(x, out, g, cfg, br_cap)
        if frac is None:
            # A BLOCK_ROWS the op's own L1 solve cannot reach on this geometry.  Recorded,
            # not asserted: it constrains baseline and candidate identically.
            MANIFEST_DIR.mkdir(exist_ok=True)
            (MANIFEST_DIR / f"{case}.jsonl").write_text("")
            print(f"  [{case}] BLOCK_ROWS cap {br_cap} UNREACHABLE by the op's L1 solve -- no launches")
            pytest.skip(f"{case}: BLOCK_ROWS <= {br_cap} unreachable (op L1 solve)")

        results = []
        outs = []

        def one(v, rep=0):
            orig = rpd.L1_SAFETY_FRACTION
            rpd.L1_SAFETY_FRACTION = frac
            try:
                y, info = bench.run(x, out, g, variant=v, epsilon=eps, compute_config=cfg)
            except RuntimeError as e:
                if "clash with L1" not in str(e) and "circular buffer" not in str(e):
                    raise
                print(f"  [{case}] {v.name:<14s} DOES NOT FIT L1 at frac={frac}")
                return None
            finally:
                rpd.L1_SAFETY_FRACTION = orig
            got = ttnn.to_torch(y)
            p = bench.pcc(got, ref)
            rr = bench.rel_rms(got, ref)
            rec = dict(
                case=case,
                rep=rep,
                name=v.name,
                layout=v.layout,
                faces=v.faces,
                zero_ablated=v.zero_ablated,
                pcc=p,
                rel_rms=rr,
                bit_exact=None,
                safety_fraction=frac,
                **info,
            )
            results.append(rec)
            outs.append((rec, got.clone()))
            print(
                f"  [{case}] {v.name:<14s} pcc={p:.6f} rel_rms={rr:.3e} "
                f"BR={info['block_rows']} nblk={info['num_blocks']} GS={info['group_size']} "
                f"txn/member/round={info['gather_txns_per_member_round']}x{info['gather_txn_bytes']}B "
                f"boot_zero={info['boot_zero_bytes']}B gatherCB={info['gather_cb_bytes']}B"
            )
            return rec

        for rep in range(repeat):
            for v in menu:
                one(v, rep)

        # ---- BIT-EXACTNESS post-pass against the baseline at the same BLOCK_ROWS ------
        base_t = next((t for r, t in outs if r["name"] == bench.BASELINE.name), None)
        for rec, t in outs:
            if base_t is not None:
                rec["bit_exact"] = bool(torch.equal(t, base_t))
            print(f"  [{case}] {rec['name']:<14s} bit_exact={rec['bit_exact']}")

        MANIFEST_DIR.mkdir(exist_ok=True)
        with (MANIFEST_DIR / f"{case}.jsonl").open("w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        # ---- the ONLY pass/fail --------------------------------------------------------
        for rec in results:
            if rec["zero_ablated"]:
                print(
                    f"  [{case}] ABLATION {rec['name']}: pcc {rec['pcc']:.6f} "
                    f"rel_rms {rec['rel_rms']:.3e} bit_exact={rec['bit_exact']} (recorded, not gated)"
                )
                continue
            assert rec["pcc"] >= PCC_GATE, f"{case}/{rec['name']}: pcc {rec['pcc']} < {PCC_GATE}"
            assert rec["rel_rms"] <= RELRMS_GATE, f"{case}/{rec['name']}: rel-RMS {rec['rel_rms']} > {RELRMS_GATE}"
            assert rec["bit_exact"] is not False, (
                f"{case}/{rec['name']}: output differs from the baseline at the same BLOCK_ROWS -- "
                f"the layout/faces change moved the ARITHMETIC, not just the addressing"
            )
    finally:
        rpd.GRID_W = grid_w_orig


@pytest.mark.parametrize("case", FOCUS_CASES)
def test_focus(device, case):
    """The full 7-point MENU on the focus geometry."""
    _body(device, case, bench.FOCUS_MENU)


def test_focus_repeat(device):
    """The full menu 3x on the focus geometry, so the load-bearing NULL calls
    (`rm_f4` vs `sm_f4` -- identical bytes, 10 transactions vs 1 -- and `rm_f2` vs
    `sm_f2`) are medians rather than single samples inside the ~2-3% noise band."""
    _body(device, "focus", bench.FOCUS_MENU, repeat=3)


@pytest.mark.parametrize("case", ["br1", "gs4_br1", "gs32"])
def test_boot_repeat(device, case):
    """3x on the three geometries whose distributed-boot call sits ON the noise band.

    `br1` / `gs4_br1` are the BLOCK_ROWS == 1, 32-round configs: single-run they read
    0.986x / 0.990x, but the strictly-LESS-work `rm_f2_nozero` read 0.998x / 0.984x on the
    same runs -- i.e. removing work "cost" 1.6% -- so the band there is ~2% and a single
    sample cannot tell a small regression from noise.  `gs32` is the biggest win (1.131x)
    and is confirmed for the same reason, in the other direction.
    """
    _body(device, case, bench.BOOT_MENU, repeat=3)


@pytest.mark.parametrize("case", SWEEP_A)
def test_sweep_a(device, case):
    """BLOCK_ROWS + small-GROUP_SIZE domain sweep, 3 points each."""
    _body(device, case, bench.SWEEP_MENU)


@pytest.mark.parametrize("case", SWEEP_B)
def test_sweep_b(device, case):
    """Wide / odd GROUP_SIZE + the interleaved path, 3 points each."""
    _body(device, case, bench.SWEEP_MENU)


def test_falsify(device):
    """THE BENCH'S OWN SELF-CHECK -- run this before trusting any number above.

    Every one of the seven menu points came out BIT-EXACT on the first attempt.  That is the
    expected result (the layout permutes pages, not arithmetic) but it is ALSO exactly what a
    silently-dead `#define` looks like: if `RMS_GATHER_LAYOUT` never reached the kernels,
    every variant would just be the baseline and every variant would be bit-exact.

    So this runs a point that MUST be wrong: the WRITER lands slot-major while the COMPUTE
    fold still reads row-major.  A pass here means the switch is live in BOTH kernels.
    """
    shape, shard, layout, _, _ = CASES["focus"]
    eps = 1e-6
    cfg = bench._perf_config()
    x_t, g_t, x, g, mc = _mk_tensors(device, shape, shard, layout)
    ref = bench.torch_reference(x_t, g_t, eps)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)

    y_ok, _ = bench.run(x, out, g, variant=bench.CANDIDATE, epsilon=eps, compute_config=cfg)
    got_ok = ttnn.to_torch(y_ok).clone()
    y_bad, _ = bench.run(x, out, g, variant=bench.FALSIFY, epsilon=eps, compute_config=cfg)
    got_bad = ttnn.to_torch(y_bad).clone()

    p_ok, p_bad = bench.pcc(got_ok, ref), bench.pcc(got_bad, ref)
    r_ok, r_bad = bench.rel_rms(got_ok, ref), bench.rel_rms(got_bad, ref)
    print(f"  [falsify] sm_f4          pcc={p_ok:.6f} rel_rms={r_ok:.3e}")
    print(f"  [falsify] sm_f4_FALSIFY  pcc={p_bad:.6f} rel_rms={r_bad:.3e}")
    assert p_ok >= PCC_GATE and r_ok <= RELRMS_GATE, "the candidate itself is not correct"
    assert not torch.equal(got_ok, got_bad), (
        "the writer/compute layout MISMATCH produced a BIT-IDENTICAL result -- "
        "RMS_GATHER_LAYOUT is not reaching the kernels and every number in this bench is void"
    )


@pytest.mark.parametrize("case", list(CASES))
def test_single(device, case):
    """One geometry, the full menu -- for bring-up / --dev triage."""
    _body(device, case, bench.FOCUS_MENU)
