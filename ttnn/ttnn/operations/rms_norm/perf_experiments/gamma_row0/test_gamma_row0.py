# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gamma_row0 isolated bake-off driver.

Correctness is the ONLY assertion here; perf is measured, never asserted.

    # 1. plans (host-only decisions: regime, row-blocks per core, CB sizes)
    scripts/run_safe_pytest.sh <this file> -k plans -s

    # 2. correctness gate for every arm x every shape (torch PCC)
    scripts/run_safe_pytest.sh <this file> -k correctness -s --run-all

    # 3. the profiled bake-off
    scripts/run_safe_pytest.sh --profile <this file> -k bench -s
    python3 -c "from ttnn.operations.rms_norm.perf_experiments.gamma_row0 import lab_bench as b; \
                b.print_report('<the printed CSV path>')"
"""

import os

import pytest
import ttnn

from ttnn.operations.rms_norm.perf_experiments.gamma_row0 import lab_bench as B

# Soft PCC gate from the focus case's feature_spec extras.
PCC_THRESHOLD = 0.9995


def _sel(var, default):
    v = os.environ.get(var)
    return [s for s in v.split(",") if s] if v else default


@pytest.mark.timeout(1200)
def test_plans(device):
    """Host-only: dump the blocking plan of every sweep shape."""
    print()
    hdr = f"{'shape':<20} {'regime':>6} {'Rt':>6} {'Wt_core':>8} {'BLOCK_HT':>9} {'WT_SCALE':>9} {'gCB pages':>10} {'ws KB':>8}"
    print(hdr)
    for name in B.SHAPES:
        p = B.plan_of(device, name)
        gpages = dict((i, n) for i, n, _, _ in p.cb_layout).get(1, 0)
        print(
            f"{name:<20} {p.regime:>6} {p.Rt:>6} {p.Wt_core:>8} {p.BLOCK_HT:>9} "
            f"{p.WT_SCALE_BLOCK:>9} {gpages:>10} {p.working_set_bytes() // 1024:>8}"
        )
        print(f"{'':<20}   num_row_blocks={p.num_row_blocks} l1_budget_KB={p.l1_cb_budget // 1024}")


@pytest.mark.timeout(3600)
def test_correctness(device):
    """Every arm on every shape, gated on torch values (never on a perf direction)."""
    import torch

    fails = []
    print()
    for name in _sel("G0_SHAPES", list(B.SHAPES)):
        x, g, xt, gt = B.make_tensors(device, name)
        cfg = B.CONFIGS[B.SHAPES[name][5]]()
        ref = B.torch_reference(xt, gt)
        for arm, levers in B.ARMS.items():
            try:
                y = B.lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=levers)
                got = torch.Tensor(ttnn.to_torch(y)).reshape(ref.shape)
                p = B.pcc(got, ref)
                amax = (got.to(torch.float32) - ref).abs().max().item()
                ok = p >= PCC_THRESHOLD and torch.isfinite(got.to(torch.float32)).all().item()
            except Exception as exc:  # a variant that cannot even run is DATA
                p, amax, ok = float("nan"), float("nan"), False
                print(f"  {name:<20} {arm:<18} RAISED {type(exc).__name__}: {exc}")
            print(f"  {name:<20} {arm:<18} pcc={p:.6f} amax={amax:.4g} {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append((name, arm, p))
    # NEGATIVE CONTROL: proves the partial read is really in effect.
    x, g, xt, gt = B.make_tensors(device, "focus")
    cfg = B.CONFIGS[B.SHAPES["focus"][5]]()
    y = B.lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=B.NEG_CONTROL)
    neg_pcc = B.pcc(torch.Tensor(ttnn.to_torch(y)).reshape(xt.shape), B.torch_reference(xt, gt))
    print(f"  {'focus':<20} {'NEG(run0 only)':<18} pcc={neg_pcc:.6f} (must be < gate)")
    # nan (a NaN output) also counts as failing the gate - that is the control firing.
    assert not (neg_pcc >= PCC_THRESHOLD), (
        "negative control PASSED: the partial gamma read is NOT in effect, "
        "so every arm is secretly measuring the baseline"
    )

    assert not fails, f"correctness failures: {fails}"


@pytest.mark.timeout(3600)
def test_bench(device):
    """Profiled bake-off: one manifest, all arms, so the device lock is paid once."""
    manifest = []
    arms = _sel("G0_ARMS", ["baseline", "span", "faces"])
    for name in _sel("G0_SHAPES", list(B.SHAPES)):
        for arm in arms:
            B.run_arm(device, manifest, name, arm)
    path = B.write_manifest(manifest)
    print(f"\nGAMMA_ROW0: manifest -> {path} ({len(manifest)} arms)")
    for a in manifest:
        print(f"  {a['label']:<40} levers={a['levers']}")
    assert manifest
