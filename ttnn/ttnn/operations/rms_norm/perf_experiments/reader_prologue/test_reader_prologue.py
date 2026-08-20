# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reader_prologue isolated bake-off driver.

Correctness is the ONLY assertion here; perf is measured, never asserted.

    # 1. host-only plans (regime, W-split group, Wt_core per shape)
    scripts/run_safe_pytest.sh <this file> -k plans -s

    # 2. correctness gate: every arm x every shape (torch PCC)
    scripts/run_safe_pytest.sh <this file> -k correctness -s --run-all

    # 3. the profiled bake-off
    scripts/run_safe_pytest.sh --profile <this file> -k bench -s
    python3 -c "from ttnn.operations.rms_norm.perf_experiments.reader_prologue import lab_bench as b; \
                b.print_report('<the printed CSV path>')"
"""

import os

import pytest
import ttnn

from ttnn.operations.rms_norm.perf_experiments.reader_prologue import lab_bench as B

# Soft PCC gate from the focus case's feature_spec extras.
PCC_THRESHOLD = 0.9995


def _sel(var, default):
    v = os.environ.get(var)
    return [s for s in v.split(",") if s] if v else default


@pytest.mark.timeout(1200)
def test_plans(device):
    print()
    hdr = f"{'shape':<16} {'regime':>6} {'Rt':>6} {'Wt_core':>8} {'BLOCK_HT':>9} {'G':>4} {'blocks':>7}"
    print(hdr)
    for name in B.SHAPES:
        p = B.plan_of(device, name)
        print(
            f"{name:<16} {p.regime:>6} {p.Rt:>6} {p.Wt_core:>8} {p.BLOCK_HT:>9} "
            f"{p.group_size:>4} {p.num_row_blocks:>7}"
        )


@pytest.mark.timeout(3600)
def test_correctness(device):
    """Every arm on every shape, gated on torch values (never on a perf direction)."""
    import torch

    fails = []
    print()
    for name in _sel("RP_SHAPES", list(B.SHAPES)):
        x, g, xt, gt = B.make_tensors(device, name)
        cfg = B.CONFIGS[B.SHAPES[name][5]]()
        ref = B.torch_reference(xt, gt)
        for arm in _sel("RP_ARMS", list(B.ARMS)):
            try:
                y = B.lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=dict(B.ARMS[arm]))
                got = torch.Tensor(ttnn.to_torch(y)).reshape(ref.shape)
                p = B.pcc(got, ref)
                amax = (got.to(torch.float32) - ref).abs().max().item()
                ok = p >= PCC_THRESHOLD and torch.isfinite(got.to(torch.float32)).all().item()
            except Exception as exc:  # a variant that cannot even run is DATA
                p, amax, ok = float("nan"), float("nan"), False
                print(f"  {name:<16} {arm:<12} RAISED {type(exc).__name__}: {exc}")
            print(f"  {name:<16} {arm:<12} pcc={p:.6f} amax={amax:.4g} {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append((name, arm, p))
    assert not fails, f"correctness failures: {fails}"


@pytest.mark.timeout(3600)
def test_bench(device):
    """Profiled bake-off: one manifest, all arms, so the device lock is paid once."""
    manifest = []
    for name in _sel("RP_SHAPES", list(B.SHAPES)):
        for arm in _sel("RP_ARMS", list(B.ARMS)):
            B.run_arm(device, manifest, name, arm)
    path = B.write_manifest(manifest)
    print(f"\nREADER_PROLOGUE: manifest -> {path} ({len(manifest)} arms)")
    assert manifest
