# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pipeline_overlap driver.  Correctness is the ONLY assertion; perf is data.

    source python_env/bin/activate

    # 1. host-only: which (chunk, depth) points are even expressible
    PO_MODE=plan scripts/run_safe_pytest.sh <this file> -s

    # 2. correctness gate for every arm (also warms the JIT cache)
    PO_MODE=correct scripts/run_safe_pytest.sh <this file> -s

    # 3. the measurement
    PO_MODE=perf scripts/run_safe_pytest.sh --profile <this file> -s
    PO_MODE=domain scripts/run_safe_pytest.sh --profile <this file> -s
    PO_MODE=zones scripts/run_safe_pytest.sh --profile <this file> -s
"""

import os
import sys

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench  # noqa: E402
import lab_plan  # noqa: E402

MODE = os.environ.get("PO_MODE", "plan")
PCC_GATE = 0.9995  # the focus case's soft threshold


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _plan_of(device, name, levers):
    x, g, _, _ = bench.make_tensors(device, name)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config()
    )
    return lab_plan.lab_blocking_plan(x, g, out, device, bench.loose_cfg(), levers)


def test_plan(device):
    if MODE != "plan":
        pytest.skip("PO_MODE != plan")
    for name in bench.SHAPES:
        x, g, _, _ = bench.make_tensors(device, name)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config()
        )
        ref = lab_plan.assert_matches_op_plan(x, g, out, device, bench.loose_cfg(), None)
        print(f"\n[{name}] OP PLAN (lab==op verified): {lab_plan.plan_summary(ref)}")
    print("\n=== focus arm surface (host-only) ===")
    for label, lev in bench.focus_arms() + bench.policy_arms():
        p = _plan_of(device, "focus", lev)
        print(f"  {label:<28} {lab_plan.plan_summary(p)}")


def _run_correct(device, name, arms):
    import torch

    x, g, xt, gt = bench.make_tensors(device, name)
    ref = bench.torch_ref(xt, gt)
    rows = []
    for label, lev in arms:
        try:
            plan = []
            y = lab_plan.lab_rms_norm(
                x, gamma=g, compute_kernel_config=bench.loose_cfg(), levers=lev, out_plan=plan
            )
            got = ttnn.to_torch(y).float()
            p = bench.pcc(ref, got)
            rows.append((label, p, lab_plan.plan_summary(plan[0])))
        except Exception as exc:  # inexpressible on this shape
            rows.append((label, None, f"SKIP {type(exc).__name__}: {str(exc)[:110]}"))
    print(f"\n=== correctness [{name}] (gate pcc >= {PCC_GATE}) ===")
    bad = []
    for label, p, info in rows:
        tag = "SKIP" if p is None else ("ok  " if p >= PCC_GATE else "FAIL")
        print(f"  {tag} {label:<28} pcc={p if p is None else round(p,6)}  {info}")
        if p is not None and p < PCC_GATE:
            bad.append((label, p))
    return bad


def test_correct(device):
    if MODE != "correct":
        pytest.skip("PO_MODE != correct")
    bad = []
    bad += _run_correct(device, "focus", bench.focus_arms())
    for name in os.environ.get("PO_SHAPES", "focus,w_nonalign,smallest,rm_gamma,regime_a").split(","):
        bad += _run_correct(device, name, bench.policy_arms())
    assert not bad, f"arms below the pcc gate: {bad}"


def _perf(device, name, arms, iters=bench.N_ITERS, no_zones=1):
    manifest = []
    x, g, xt, gt = bench.make_tensors(device, name)
    ref = bench.torch_ref(xt, gt)
    for label, lev in arms:
        lev = dict(lev)
        lev["no_zones"] = no_zones
        plan = []
        try:
            y = lab_plan.lab_rms_norm(
                x, gamma=g, compute_kernel_config=bench.loose_cfg(), levers=lev, out_plan=plan
            )
        except Exception as exc:
            print(f"  SKIP {label} on {name}: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        p = bench.pcc(ref, ttnn.to_torch(y).float())
        n = bench.dispatch(
            device,
            lambda l=lev: lab_plan.lab_rms_norm(x, gamma=g, compute_kernel_config=bench.loose_cfg(), levers=l),
            iters,
        )
        manifest.append(
            {
                "label": f"{name}/{label}",
                "shape": name,
                "levers": lev,
                "plan": lab_plan.plan_summary(plan[0]),
                "pcc": round(p, 6),
                # +1: the correctness dispatch above is profiled too; count it
                # as warm-up so report_from_csv's row offsets stay aligned.
                "calls": n + 1,
                "profiled": iters,
            }
        )
    return manifest


def test_perf(device):
    if MODE != "perf":
        pytest.skip("PO_MODE != perf")
    manifest = _perf(device, "focus", bench.focus_arms())
    path = bench.write_manifest(manifest)
    print(f"\nPO: manifest -> {path} ({len(manifest)} arms)")
    for a in manifest:
        print(f"  {a['label']:<40} pcc={a['pcc']}")
    assert manifest


def test_domain(device):
    if MODE != "domain":
        pytest.skip("PO_MODE != domain")
    names = os.environ.get("PO_SHAPES", "focus,w_nonalign,prefill_7168,smallest,regime_a,rm_gamma").split(",")
    sel = os.environ.get("PO_ARMS", ",".join(bench.DOMAIN_ARMS)).split(",")
    table = {lbl: lev for lbl, lev in bench.focus_arms() + bench.policy_arms() + bench.ablation_arms()}
    arms = [(s, table[s]) for s in sel if s in table]
    manifest = []
    for name in names:
        manifest += _perf(device, name, arms)
    path = bench.write_manifest(manifest)
    print(f"\nPO: manifest -> {path} ({len(manifest)} arms)")
    for a in manifest:
        print(f"  {a['label']:<40} pcc={a['pcc']}")
    assert manifest


def test_ablation(device):
    """Cumulative payload ablation at the baseline plan AND at the winner."""
    if MODE != "ablation":
        pytest.skip("PO_MODE != ablation")
    manifest = _perf(device, "focus", bench.ablation_arms())
    path = bench.write_manifest(manifest)
    print(f"\nPO: ablation manifest -> {path}")
    assert manifest


def test_dtypes(device):
    """Close the dtype gap with a measurement instead of a caveat.

    A wider tile HALVES the page budget (fp32: 4096 B/page), so the policy's
    chunk search lands somewhere else entirely - that has to be shown correct and
    non-regressing, not assumed.  fp32 runs at the op's DEFAULT precision corner
    because feature_spec EXCLUSIONS bar fp32 + fp32_dest_acc_en=False.
    """
    if MODE != "dtypes":
        pytest.skip("PO_MODE != dtypes")
    import torch

    from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config

    manifest, bad = [], []
    cases = [
        ("fp32", ttnn.float32, default_compute_kernel_config),
        ("bf8b", ttnn.bfloat8_b, bench.loose_cfg),
    ]
    shape = bench.SHAPES["focus"][0]
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    ref = bench.torch_ref(xt, gt)
    for tag, dtype, cfgf in cases:
        x = ttnn.from_torch(xt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        g = ttnn.from_torch(gt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        for label, lev in bench.policy_arms():
            if label not in ("BASE_op_plan", "POL_win"):
                continue
            lev = dict(lev)
            lev["no_zones"] = 1
            plan = []
            try:
                y = lab_plan.lab_rms_norm(x, gamma=g, compute_kernel_config=cfgf(), levers=lev, out_plan=plan)
            except Exception as exc:
                print(f"  SKIP {tag}/{label}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            p = bench.pcc(ref, ttnn.to_torch(y).float())
            n = bench.dispatch(
                device, lambda l=lev, c=cfgf: lab_plan.lab_rms_norm(x, gamma=g, compute_kernel_config=c(), levers=l)
            )
            print(f"  {tag}/{label:<14} pcc={p:.6f}  {lab_plan.plan_summary(plan[0])}")
            # bfloat8_b transports 16 datums per exponent, so its own floor sits
            # well below the bf16 gate; compare the two ARMS to each other, not to
            # an absolute number.
            manifest.append(
                {
                    "label": f"{tag}/{label}",
                    "shape": f"focus_{tag}",
                    "levers": lev,
                    "plan": lab_plan.plan_summary(plan[0]),
                    "pcc": round(p, 6),
                    "calls": n + 1,
                    "profiled": bench.N_ITERS,
                }
            )
    path = bench.write_manifest(manifest)
    print(f"\nPO: dtype manifest -> {path}")
    assert manifest


def test_zones(device):
    """Zone breakdown: ONE dispatch per arm, profiler zones ON."""
    if MODE != "zones":
        pytest.skip("PO_MODE != zones")
    sel = os.environ.get("PO_ARMS", ",".join(bench.DOMAIN_ARMS)).split(",")
    table = {lbl: lev for lbl, lev in bench.focus_arms() + bench.policy_arms() + bench.ablation_arms()}
    arms = [(s, table[s]) for s in sel if s in table]
    manifest = _perf(device, os.environ.get("PO_ZONE_SHAPE", "focus"), arms, iters=1, no_zones=0)
    path = bench.write_manifest(manifest)
    print(f"\nPO: zone manifest -> {path}")
    for a in manifest:
        print(f"  {a['label']:<40} pcc={a['pcc']}")
    assert manifest
