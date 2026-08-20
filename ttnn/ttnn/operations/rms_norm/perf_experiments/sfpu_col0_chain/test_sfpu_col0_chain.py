# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sfpu_col0_chain WHOLE-OP bake-off driver.  Correctness is the ONLY assertion.

    # 1. plans (host-only)
    scripts/run_safe_pytest.sh <this file> -k plans -s

    # 2. correctness + don't-care poison probes, every arm x every shape
    scripts/run_safe_pytest.sh <this file> -k correctness -s --run-all

    # 3. the profiled bake-off
    scripts/run_safe_pytest.sh --profile <this file> -k bench -s
    python3 -c "from ttnn.operations.rms_norm.perf_experiments.sfpu_col0_chain import lab_bench as b; \
                b.print_report('<the printed CSV path>')"
"""

import os

import pytest
import ttnn

from ttnn.operations.rms_norm.perf_experiments.sfpu_col0_chain import lab_bench as B

# Soft PCC gate from the focus case's feature_spec extras.
PCC_THRESHOLD = 0.9995


def _sel(var, default):
    v = os.environ.get(var)
    return [s for s in v.split(",") if s] if v else default


@pytest.mark.timeout(1200)
def test_plans(device):
    print()
    print(f"{'shape':<20} {'regime':>6} {'Rt':>6} {'Wt_core':>8} {'BLOCK_HT':>9} {'G':>3} {'rowblks':>8}")
    for name in B.SHAPES:
        p = B.plan_of(device, name)
        print(
            f"{name:<20} {p.regime:>6} {p.Rt:>6} {p.Wt_core:>8} {p.BLOCK_HT:>9} "
            f"{p.group_size:>3} {p.num_row_blocks:>8}"
        )


@pytest.mark.timeout(5400)
def test_correctness(device):
    """Every arm on every shape, gated on torch values (never on a perf direction)."""
    import torch

    fails = []
    print()
    print(f"  {'shape':<20} {'arm':<14} {'pcc':>10} {'row-scale bias':>15} {'amax':>10}")
    for name in _sel("SC_SHAPES", list(B.SHAPES)):
        x, g, xt, gt = B.make_tensors(device, name)
        cfg = B.CONFIGS[B.SHAPES[name][5]]()
        ref = B.torch_reference(xt, gt)
        for arm in _sel("SC_ARMS", list(B.ARMS) + list(B.POISON_ARMS)):
            try:
                y = B.lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=B.levers_for(name, arm))
                got = torch.Tensor(ttnn.to_torch(y)).reshape(ref.shape)
                p = B.pcc(got, ref)
                bias = B.row_scale_bias(got, ref)
                amax = (got.to(torch.float32) - ref).abs().max().item()
                ok = p >= PCC_THRESHOLD and torch.isfinite(got.to(torch.float32)).all().item()
            except Exception as exc:
                p, bias, amax, ok = float("nan"), float("nan"), float("nan"), False
                print(f"  {name:<20} {arm:<14} RAISED {type(exc).__name__}: {exc}")
            print(f"  {name:<20} {arm:<14} {p:>10.6f} {bias:>14.3f}% {amax:>10.4g} {'' if ok else 'FAIL'}")
            if not ok:
                fails.append((name, arm, p))
    # POSITIVE CONTROL for the poison probe: the same stamp WITH column 0 must
    # break the output.  If it does not, the probe lands nowhere and the two
    # poison arms above proved nothing.
    name = _sel("SC_SHAPES", list(B.SHAPES))[0]
    x, g, xt, gt = B.make_tensors(device, name)
    cfg = B.CONFIGS[B.SHAPES[name][5]]()
    y = B.lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=B.levers_for(name, "poison_all"))
    ctrl = B.pcc(torch.Tensor(ttnn.to_torch(y)).reshape(xt.shape), B.torch_reference(xt, gt))
    print(f"  {name:<20} {'poison_all':<14} {ctrl:>10.6f}  (POSITIVE CONTROL - must be < gate)")
    assert not (ctrl >= PCC_THRESHOLD), (
        "positive control PASSED: the NaN poison never reaches the consumer, so the "
        "poison_cskip / poison_base arms are vacuous"
    )

    assert not fails, f"correctness failures: {fails}"


@pytest.mark.timeout(7200)
def test_bench(device):
    """Profiled bake-off: one manifest, all arms, so the device lock is paid once."""
    manifest = []
    arms = _sel("SC_ARMS", list(B.ARMS))
    for name in _sel("SC_SHAPES", list(B.SHAPES)):
        for arm in arms:
            B.run_arm(device, manifest, name, arm)
    path = B.write_manifest(manifest)
    print(f"\nSFPU_COL0_CHAIN: manifest -> {path} ({len(manifest)} arms)")
    assert manifest
