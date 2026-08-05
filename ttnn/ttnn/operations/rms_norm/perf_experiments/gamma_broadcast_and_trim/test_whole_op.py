# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Whole-op bake-off for `gamma_broadcast_and_trim` on the CLONED op (descriptor.py +
# kernels/), never the real one.  Correctness (pcc vs torch) is the ONLY pass/fail;
# every ns comes from `scripts/run_safe_pytest.sh --profile` + the emitted
# `ops_perf_results_*.csv`, matched to a launch through `perf_manifest.csv`.
#
# The user config is FROZEN for every variant of every case (HiFi2 /
# fp32_dest_acc_en=False / math_approx_mode=False / bf16), so no number here can be
# bought with precision.

import csv
import os

import pytest
import ttnn

from ttnn.operations.rms_norm.perf_experiments.gamma_broadcast_and_trim import whole_op
from ttnn.operations.rms_norm.perf_experiments.gamma_broadcast_and_trim import descriptor as _desc


# `torch` is imported LAZILY here, on first attribute access, rather than at module scope.
# `ttnn/ttnn/operations/__init__.py` runs pkgutil.walk_packages over the operations tree, and
# the repo's `check-torch-imports-in-ttnn` pre-commit hook forbids a global torch import
# anywhere under ttnn/ for exactly that reason.  See perf_experiments/README.md -- an
# __init__.py in that directory once broke `import ttnn` repo-wide twice in one round.
# Every `torch.<attr>` use below is unchanged; the proxy just defers the import.
class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


_ML = ttnn.TensorMemoryLayout
_HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(_HERE, "perf_manifest.csv")

PCC_GATE = 0.9995  # the focus case's soft gate — reported, and asserted at 0.999


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    if torch.isnan(a).any() or torch.isinf(a).any():
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _golden(x, gamma, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(1, 1, 1, -1)
    return out


def _note(tag, extra=""):
    with open(MANIFEST, "a") as f:
        csv.writer(f).writerow([tag, extra])


# ======================================================================================
# CASES
# ======================================================================================
INTERLEAVED_CASES = [
    pytest.param((1, 1, 8192, 1024), None, None, id="il_w1024"),  # PRIMARY, baseline 104705
    pytest.param((1, 1, 8192, 2304), None, None, id="il_w2304"),  # baseline 222536
    pytest.param((1, 1, 8192, 5120), None, None, id="il_w5120"),  # baseline 475772 (control)
    pytest.param((1, 1, 8192, 7168), None, None, id="il_w7168"),  # baseline 641683 (control)
]
FOCUS_CASE = [
    pytest.param((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, id="bshard_64c"),  # 64753
]

MENU = ["baseline", "trim_half", "trim_rows", "trim_rows64", "mcast", "mcast_trim_rows", "mcast_trim_half"]


def _make(
    device,
    shape,
    shard,
    memory_layout,
    *,
    gamma_layout=ttnn.TILE_LAYOUT,
    has_gamma=True,
    poison=0.0,
    gamma_dtype=ttnn.bfloat16,
):
    from eval.sharding import shard_config

    torch.manual_seed(42)
    W = shape[-1]
    mc = None
    if shard is not None:
        mc = shard_config(
            shard[0], shard[1], memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
    xt = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    gt = None
    g = None
    if has_gamma:
        gt = torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W)
        if gamma_dtype == ttnn.bfloat8_b:
            # bfloat8_b round-trips through the tilizer; compare against what the device
            # actually holds, not against the fp32 draw.
            g = ttnn.from_torch(gt, dtype=gamma_dtype, layout=gamma_layout, device=device)
            gt = ttnn.to_torch(g).reshape(1, 1, 1, W)
            return x, xt, g, gt, mc
        if poison != 0.0:
            # THE SEEDED-WRONG-ROWS EXPERIMENT.  A (1,1,32,W) gamma has the SAME tile
            # pages as a (1,1,1,W) one (Wt tiles of tile-row 0), so the reader reads it
            # byte-identically — but now rows 1..31 carry INDEPENDENT garbage five orders
            # of magnitude larger than the real weights instead of the tilizer's zeros.
            # If `mul<BroadcastDim::Row>` reads any row but row 0, the output is wrecked.
            full = gt.reshape(1, 1, 1, W).expand(1, 1, 32, W).clone()
            if poison < 0:
                # THE TEETH CONTROL: corrupt ROW 0 and leave 1..31 clean.  pcc MUST
                # collapse — otherwise this harness is not feeding the op at all.
                full[:, :, 0, :] = torch.randn(W, dtype=torch.bfloat16) * 1e4
            else:
                full[:, :, 1:, :] = torch.randn(31, W, dtype=torch.bfloat16) * poison
            g = ttnn.from_torch(full, dtype=gamma_dtype, layout=gamma_layout, device=device)
        else:
            g = ttnn.from_torch(gt, dtype=gamma_dtype, layout=gamma_layout, device=device)
    return x, xt, g, gt, mc


def _run_and_check(device, shape, shard, memory_layout, option_name, *, gamma_layout=ttnn.TILE_LAYOUT, has_gamma=True):
    x, xt, g, gt, mc = _make(device, shape, shard, memory_layout, gamma_layout=gamma_layout, has_gamma=has_gamma)
    opt = whole_op.OPTIONS[option_name]
    out = whole_op.run(x, gamma=g, trim=opt["trim"], mcast=opt["mcast"], memory_config=mc)
    got = ttnn.to_torch(out)
    pcc = _pcc(got, _golden(xt, gt))
    print(
        f"\nPCC[{option_name}] shape={tuple(shape)} shard={shard} gamma_rm={gamma_layout != ttnn.TILE_LAYOUT} "
        f"has_gamma={has_gamma} -> {pcc:.7f}  (soft gate {PCC_GATE})  PLAN[{_desc.GAMMA_PLAN_NOTE[0]}]"
    )
    assert pcc > 0.999, f"{option_name} on {shape}: pcc {pcc}"
    return pcc


# ======================================================================================
# (0) THE CLONE IS THE BASELINE.  With both knobs 0 the clone must match the real op's
#     output exactly — this is what makes every delta below attributable.
# ======================================================================================
def test_clone_is_the_op(device):
    from ttnn.operations.rms_norm import rms_norm

    shape = (1, 1, 512, 1024)
    x, xt, g, gt, _ = _make(device, shape, None, None)
    ref = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=whole_op.compute_config()))
    got = ttnn.to_torch(whole_op.run(x, gamma=g, trim=0, mcast=0))
    assert torch.equal(ref, got), "the clone diverged from the real op at knobs (0,0)"


# ======================================================================================
# (1) THE SAFETY EXPERIMENT for the trim: does `mul<BroadcastDim::Row>` read row 0 only?
# ======================================================================================
@pytest.mark.parametrize("poison", [0.0, 1e5, -1.0], ids=["clean", "poison_rows1to31", "poison_row0_TEETH"])
@pytest.mark.parametrize("shape", [(1, 1, 512, 1024), (1, 1, 512, 2304)], ids=["w1024", "w2304"])
def test_row0_only(device, shape, poison):
    """The op's real consumer, over a gamma whose tile rows 1..31 are seeded FIVE ORDERS
    OF MAGNITUDE wrong.  A clean pcc proves those rows are never read — which is exactly
    the licence the face-row trim needs."""
    x, xt, g, gt, _ = _make(device, shape, None, None, poison=poison)
    got = ttnn.to_torch(whole_op.run(x, gamma=g, trim=0, mcast=0))
    pcc = _pcc(got, _golden(xt, gt))
    print(f"\nROW0_ONLY shape={tuple(shape)} poison={poison:g} -> pcc {pcc:.7f}")
    if poison == 0.0:
        assert pcc > 0.999
    elif poison < 0:
        assert not (pcc > 0.999), f"TEETH control did not fire (pcc {pcc}) — the harness is not feeding gamma"
    _note(f"row0_only_{shape[-1]}_{poison:g}", f"pcc={pcc}")


# ======================================================================================
# (1b) BIT-EXACTNESS.  Both sub-ideas only MOVE BYTES — they change neither the math nor
#      its precision — so every option must be torch.equal to the baseline, not merely
#      close.  Anything less would be a red flag, not a trade.
# ======================================================================================
@pytest.mark.parametrize("option", [o for o in MENU if o != "baseline"])
@pytest.mark.parametrize(
    "shape, shard, memory_layout",
    [
        pytest.param((1, 1, 8192, 1024), None, None, id="il_w1024"),
        pytest.param((1, 1, 8192, 2304), None, None, id="il_w2304"),
        pytest.param((1, 1, 8192, 7168), None, None, id="il_w7168"),
        pytest.param((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, id="bshard_64c"),
        pytest.param((1, 1, 224, 3072), None, None, id="il_prime_rt7"),
        pytest.param((1, 1, 32, 1024), None, None, id="il_rt1"),
        pytest.param((1, 1, 512, 1000), None, None, id="il_partial_w"),
    ],
)
def test_bit_exact_vs_baseline(device, shape, shard, memory_layout, option):
    x, xt, g, gt, mc = _make(device, shape, shard, memory_layout)
    ref = ttnn.to_torch(whole_op.run(x, gamma=g, trim=0, mcast=0, memory_config=mc))
    opt = whole_op.OPTIONS[option]
    got = ttnn.to_torch(whole_op.run(x, gamma=g, trim=opt["trim"], mcast=opt["mcast"], memory_config=mc))
    assert torch.equal(ref, got), f"{option} on {shape}/{shard} is NOT bit-exact vs baseline"


# ======================================================================================
# (2) THE OPTION MENU, whole-op, on the interleaved prefill targets.  ONE launch each.
# ======================================================================================
@pytest.mark.parametrize("option", MENU)
@pytest.mark.parametrize("shape, shard, memory_layout", INTERLEAVED_CASES)
def test_pm_interleaved(device, shape, shard, memory_layout, option):
    _note(f"il_W{shape[-1]}", option)
    _run_and_check(device, shape, shard, memory_layout, option)


# ======================================================================================
# (3) FOCUS-SHAPE NO-REGRESSION CHECK (BLOCK_SHARDED 64c — where gamma is 3.8% of wall).
# ======================================================================================
@pytest.mark.parametrize("option", MENU)
@pytest.mark.parametrize("shape, shard, memory_layout", FOCUS_CASE)
def test_pm_focus(device, shape, shard, memory_layout, option):
    _note("bshard_64c", option)
    _run_and_check(device, shape, shard, memory_layout, option)


# ======================================================================================
# (4) CONTROLS — no gamma, and a ROW_MAJOR gamma (where the trim is inexpressible).
# ======================================================================================
@pytest.mark.parametrize("option", ["baseline"])
def test_pm_nogamma(device, option):
    _note("il_W1024_nogamma", option)
    _run_and_check(device, (1, 1, 8192, 1024), None, None, option, has_gamma=False)


# ======================================================================================
# (2b) PRIMARY TARGET ONLY, all options + the no-gamma control, in ONE short profiled
#      run — few enough launches that `probes/zone_breakdown.py` can attribute the delta
#      to `reader_read_gamma` without the 125-entry-per-RISC zone budget truncating.
# ======================================================================================
@pytest.mark.parametrize("option", MENU + ["nogamma"])
def test_pz_primary(device, option):
    _note("il_W1024", option)
    if option == "nogamma":
        _run_and_check(device, (1, 1, 8192, 1024), None, None, "baseline", has_gamma=False)
    else:
        _run_and_check(device, (1, 1, 8192, 1024), None, None, option)


# ======================================================================================
# (5) THE SHARING-GROUP SWEEP.  In the row-split regime the number of cores IS the gamma
#     sharing group, and `split_work_to_cores` sets it to min(Rt, 110) -- so choosing Rt
#     chooses the group size directly, in the WHOLE OP rather than in a stage mock.
#     Rt values are multiples of the grid width (11) so the assignment stays a rectangle
#     (a ragged one is `inexpressible` for the emitters, not slow).  One tile-row per
#     core throughout, so per-core work is constant across the sweep.
# ======================================================================================
GROUP_SWEEP = [pytest.param(rt, id=f"g{rt}") for rt in (4, 8, 11, 22, 44, 55, 110)]


@pytest.mark.parametrize("option", ["baseline", "trim_rows", "mcast"])
@pytest.mark.parametrize("rt", GROUP_SWEEP)
def test_pm_group(device, rt, option):
    _note(f"group{rt:03d}", option)
    _run_and_check(device, (1, 1, 32 * rt, 1024), None, None, option)


# ======================================================================================
# (6) WIDTH (Wt) EXTREMES at the full 110-core group.
# ======================================================================================
@pytest.mark.parametrize("option", ["baseline", "trim_rows", "mcast", "mcast_trim_half"])
@pytest.mark.parametrize("W", [128, 512], ids=["wt4", "wt16"])
def test_pm_narrow(device, W, option):
    _note(f"il_W{W}", option)
    _run_and_check(device, (1, 1, 8192, W), None, None, option)


@pytest.mark.parametrize("option", ["baseline", "trim_rows", "mcast"])
def test_pm_gammarm(device, option):
    """A ROW_MAJOR gamma is a single W-element stick with NO tile padding: the host gates
    both knobs off, so all three cells must land on the SAME kernel and the same ns."""
    _note("il_W1024_gamma_rm", option)
    _run_and_check(device, (1, 1, 8192, 1024), None, None, option, gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


# ======================================================================================
# (7) GAMMA DTYPE bit-exactness.  The trim's face stride is only 64 B DRAM-aligned when
#     the tile is a LINEAR format (bf16 2048 B, fp32 4096 B); bfloat8_b's 1088 B tile has
#     a 272 B face, so the host demotes TRIM 2/3 to the half-page TRIM 1.  Measured here
#     rather than argued.
# ======================================================================================
@pytest.mark.parametrize("option", [o for o in MENU if o != "baseline"])
@pytest.mark.parametrize("gamma_dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["gamma_fp32", "gamma_bfp8"])
def test_bit_exact_gamma_dtype(device, gamma_dtype, option):
    shape = (1, 1, 1024, 1024)
    x, xt, g, gt, mc = _make(device, shape, None, None, gamma_dtype=gamma_dtype)
    ref = ttnn.to_torch(whole_op.run(x, gamma=g, trim=0, mcast=0))
    print(f"\nPLAN_REF[{gamma_dtype}] {_desc.GAMMA_PLAN_NOTE[0]}")
    opt = whole_op.OPTIONS[option]
    got = ttnn.to_torch(whole_op.run(x, gamma=g, trim=opt["trim"], mcast=opt["mcast"]))
    print(f"PLAN[{option}][{gamma_dtype}] {_desc.GAMMA_PLAN_NOTE[0]}  pcc={_pcc(got, _golden(xt, gt)):.7f}")
    assert torch.equal(ref, got), f"{option} @ gamma {gamma_dtype} is NOT bit-exact"
