# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""resident_single_read driver.  Correctness is the ONLY assertion; perf is data.

    source python_env/bin/activate

    # 1. host-only: which regime each shape lands in, and the C plan surface
    RSR_MODE=plan scripts/run_safe_pytest.sh <this file> -s

    # 2. correctness gate for every arm (also warms the JIT cache)
    RSR_MODE=correct scripts/run_safe_pytest.sh <this file> -s

    # 3. the focus measurement (chunk x depth surface)
    RSR_MODE=perf scripts/run_safe_pytest.sh --profile <this file> -s

    # 4. the domain sweep (BASE vs C_auto on every shape)
    RSR_MODE=domain scripts/run_safe_pytest.sh --profile <this file> -s

    # 5. precision matrix (PCC + row-scale bias, BASE vs C, Wt 32/64/128/224)
    RSR_MODE=precision scripts/run_safe_pytest.sh <this file> -s

    # 6. per-stage zone breakdown (zones ON, one dispatch per arm)
    RSR_MODE=zones scripts/run_safe_pytest.sh --profile <this file> -s
"""

import os
import sys

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rsr_bench as bench  # noqa: E402  (UNIQUE module name: ttnn.operations auto-imports
#     every .py under the operations tree, so a bare `bench` would collide with a
#     sibling experiment dir that also has one - it did, first run.)
import rsr_plan  # noqa: E402

MODE = os.environ.get("RSR_MODE", "plan")
PCC_GATE = 0.9995  # the focus case's soft threshold


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _plan(device, name, levers, config="loose"):
    x, g, _, _ = bench.make_tensors(device, name)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config()
    )
    return rsr_plan.lab_blocking_plan(x, g, out, device, bench.CONFIGS[config](), levers)


def test_plan(device):
    if MODE != "plan":
        pytest.skip("RSR_MODE != plan")
    print("\n=== per-shape regime, baseline (allow_c=0) vs candidate ===")
    for name in bench.SHAPES:
        cfg = "default" if name == "fp32_7168" else "loose"
        x, g, _, _ = bench.make_tensors(device, name)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config()
        )
        rsr_plan.assert_matches_op_plan(x, g, out, device, bench.CONFIGS[cfg]())
        b = _plan(device, name, dict(allow_c=0), cfg)
        c = _plan(device, name, dict(), cfg)
        print(f"\n[{name}] cfg={cfg}")
        print(f"   op/BASE : {rsr_plan.plan_summary(b)}")
        print(f"   cand    : {rsr_plan.plan_summary(c)}")
    print("\n=== focus arm surface (host-only) ===")
    for label, lev in bench.focus_arms():
        try:
            print(f"  {label:<24} {rsr_plan.plan_summary(_plan(device, 'focus', lev))}")
        except (AssertionError, ValueError) as exc:
            print(f"  {label:<24} INEXPRESSIBLE: {exc}")


def _run(device, name, lev, config="loose"):
    x, g, xt, gt = bench.make_tensors(device, name)
    plan = []
    y = rsr_plan.lab_rms_norm(
        x, gamma=g, compute_kernel_config=bench.CONFIGS[config](), levers=lev, out_plan=plan
    )
    return y, plan[0], x, g, xt, gt


def _correct(device, name, arms, config="loose"):
    ref = None
    rows, bad = [], []
    for label, lev in arms:
        try:
            y, plan, x, g, xt, gt = _run(device, name, lev, config)
            if ref is None:
                ref = bench.torch_ref(xt, gt)
            got = ttnn.to_torch(y).float()
            p = bench.pcc(ref, got)
            rows.append((label, p, rsr_plan.plan_summary(plan)))
            if p < PCC_GATE:
                bad.append((name, label, p))
        except (AssertionError, ValueError) as exc:
            rows.append((label, None, f"SKIP {type(exc).__name__}: {str(exc)[:100]}"))
    print(f"\n=== correctness [{name}] (gate pcc >= {PCC_GATE}) ===")
    for label, p, info in rows:
        tag = "SKIP" if p is None else ("ok  " if p >= PCC_GATE else "FAIL")
        print(f"  {tag} {label:<24} pcc={p if p is None else round(p, 6)}  {info}")
    return bad


def test_correct(device):
    if MODE != "correct":
        pytest.skip("RSR_MODE != correct")
    bad = []
    names = os.environ.get("RSR_SHAPES", "focus").split(",")
    for name in names:
        bad += _correct(device, name, bench.focus_arms())
    assert not bad, f"arms below the pcc gate: {bad}"


def test_correct_domain(device):
    if MODE != "correct_domain":
        pytest.skip("RSR_MODE != correct_domain")
    bad = []
    arms = [(l, dict(v)) for l, v in bench.focus_arms() if l in bench.DOMAIN_ARMS]
    for name in bench.SHAPES:
        cfg = "default" if name == "fp32_7168" else "loose"
        bad += _correct(device, name, arms, cfg)
    assert not bad, f"arms below the pcc gate: {bad}"


def _perf(device, name, arms, iters=0, no_zones=1, config="loose"):
    iters = iters or bench.N_ITERS
    manifest = []
    for label, lev in arms:
        lev = dict(lev)
        lev["no_zones"] = no_zones
        try:
            y, plan, x, g, xt, gt = _run(device, name, lev, config)
        except (AssertionError, ValueError) as exc:
            print(f"  SKIP {name}/{label}: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        p = bench.pcc(bench.torch_ref(xt, gt), ttnn.to_torch(y).float())
        n = bench.dispatch(
            device,
            lambda l=lev, xx=x, gg=g: rsr_plan.lab_rms_norm(
                xx, gamma=gg, compute_kernel_config=bench.CONFIGS[config](), levers=l
            ),
            iters,
        )
        manifest.append(
            {
                "label": f"{name}/{label}",
                "shape": name,
                "levers": lev,
                "plan": rsr_plan.plan_summary(plan),
                "pcc": round(p, 6),
                # +1 for the CORRECTNESS dispatch above: report_from_csv folds the
                # per-op CSV by dispatch ORDER and skips (calls - profiled) rows per
                # arm, so every dispatch this arm made has to be counted or every
                # LATER arm's window slides (measured: it did, first perf run).
                "calls": n + 1,
                "profiled": iters,
            }
        )
        print(f"  ran {name}/{label:<24} pcc={round(p,6)}  {rsr_plan.plan_summary(plan)}")
    return manifest


def test_perf(device):
    if MODE != "perf":
        pytest.skip("RSR_MODE != perf")
    names = os.environ.get("RSR_SHAPES", "focus").split(",")
    manifest = []
    for name in names:
        manifest += _perf(device, name, bench.focus_arms())
    path = bench.write_manifest(manifest)
    print(f"\nRSR: manifest -> {path} ({len(manifest)} arms)")
    assert manifest


def test_domain(device):
    if MODE != "domain":
        pytest.skip("RSR_MODE != domain")
    names = os.environ.get("RSR_SHAPES", ",".join(bench.SHAPES)).split(",")
    sel = os.environ.get("RSR_ARMS", ",".join(bench.DOMAIN_ARMS)).split(",")
    table = {l: v for l, v in bench.focus_arms()}
    arms = [(s, table[s]) for s in sel if s in table]
    manifest = []
    for name in names:
        cfg = "default" if name == "fp32_7168" else "loose"
        manifest += _perf(device, name, arms, config=cfg)
    path = bench.write_manifest(manifest)
    print(f"\nRSR: manifest -> {path} ({len(manifest)} arms)")
    assert manifest


def test_precision(device):
    """PCC + the row-scale bias (mean(computed_rms/reference_rms) - 1), BASE vs C.

    Both arms run the SAME frozen config (bf16 / HiFi2 / fp32_dest_acc_en=False),
    so any delta here is the reduce DATAPATH, not a precision knob.
    """
    if MODE != "precision":
        pytest.skip("RSR_MODE != precision")
    names = os.environ.get("RSR_SHAPES", "decode_1024,prec_wt64,decode_4096,focus,wide_16384").split(",")
    arms = [
        ("BASE", dict(allow_c=0)),
        ("C_auto", dict()),
        ("C_nofuse", dict(c_fused_reduce=0)),
        ("BASE_reducetile", dict(allow_c=0, reduce_via_add=0)),
        # On a width where the op already picks Regime A, force B: that is the
        # same fused-vs-streaming reduce comparison the wide shapes make, at a
        # width where both plans exist.
        ("FORCE_B", dict(allow_c=0, force_regime="B")),
    ]
    print(f"\n=== precision matrix (fp32_dest_acc_en=False, HiFi2) ===")
    for name in names:
        wt = (bench.SHAPES[name][0][-1] + 31) // 32
        print(f"\n[{name}] Wt={wt}")
        for label, lev in arms:
            try:
                y, plan, x, g, xt, gt = _run(device, name, lev)
            except (AssertionError, ValueError) as exc:
                print(f"   {label:<18} SKIP {str(exc)[:80]}")
                continue
            got = ttnn.to_torch(y).float()
            p = bench.pcc(bench.torch_ref(xt, gt), got)
            bias = bench.row_scale_bias(got, xt, gt)
            print(f"   {label:<18} pcc={p:.6f}  rms_bias={bias * 100:+.3f}%  regime={plan.regime} rva={plan.reduce_via_add}")


def test_zones(device):
    if MODE != "zones":
        pytest.skip("RSR_MODE != zones")
    sel = os.environ.get("RSR_ARMS", "BASE,C_auto").split(",")
    table = {l: v for l, v in bench.focus_arms()}
    arms = [(s, table[s]) for s in sel if s in table]
    names = os.environ.get("RSR_SHAPES", "focus").split(",")
    manifest = []
    for name in names:
        manifest += _perf(device, name, arms, iters=1, no_zones=0)
    path = bench.write_manifest(manifest)
    print(f"\nRSR: zone manifest -> {path}")
    assert manifest
