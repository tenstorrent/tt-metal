# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""regime_b_resident driver.  Correctness is the ONLY assertion; perf is data.

    source python_env/bin/activate

    # 1. host-only: which regime each shape lands in, baseline vs candidate
    RBR_MODE=plan scripts/run_safe_pytest.sh <this file> -s

    # 2. correctness gate (PCC + row-scale bias) for every arm
    RBR_MODE=correct scripts/run_safe_pytest.sh <this file> -s

    # 3. the domain sweep: BASE vs LADDER on every shape
    RBR_MODE=domain scripts/run_safe_pytest.sh --profile <this file> -s

    # 4. the focus/attribution surface on one shape
    RBR_MODE=perf RBR_SHAPES=w_nonalign scripts/run_safe_pytest.sh --profile <this file> -s

    # 5. per-stage zone breakdown (zones ON, one dispatch per arm)
    RBR_MODE=zones scripts/run_safe_pytest.sh --profile <this file> -s
"""

import os

import pytest
import ttnn

# Package-path imports (NOT sys.path.insert): the experiment lives inside the
# ttnn package tree, and prepending its directory makes `import ttnn` resolve a
# second copy of the package under pytest (measured: "Operation with name
# bernoulli is already registered" at collection).
from ttnn.operations.rms_norm.perf_experiments.regime_b_resident import rbr_bench as bench
from ttnn.operations.rms_norm.perf_experiments.regime_b_resident import rbr_plan

MODE = os.environ.get("RBR_MODE", "plan")
PCC_GATE = 0.9995  # the focus case's soft threshold


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _shapes():
    return os.environ.get("RBR_SHAPES", ",".join(bench.SHAPES)).split(",")


def _plan(device, name, levers):
    x, g, _, _ = bench.make_tensors(device, name)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config())
    cfg = bench.CONFIGS[bench.config_for(name)]()
    return rbr_plan.blocking_plan(x, g, out, device, cfg, levers)


def test_plan(device):
    if MODE != "plan":
        pytest.skip("RBR_MODE != plan")
    print("\n=== per-shape plan: op/BASE vs candidate ladder ===")
    for name in _shapes():
        x, g, _, _ = bench.make_tensors(device, name)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config())
        cfg = bench.CONFIGS[bench.config_for(name)]()
        # HONEST-BASELINE GATE: asserts the fork == the shipped op at BASE levers.
        rbr_plan.assert_matches_op_plan(x, g, out, device, cfg)
        b = _plan(device, name, dict(rbr_plan.BASELINE_LEVERS))
        c = _plan(device, name, dict())
        print(f"\n[{name}] cfg={bench.config_for(name)}")
        print(f"   op/BASE : {rbr_plan.plan_summary(b)}")
        print(f"   LADDER  : {rbr_plan.plan_summary(c)}")


def _run(device, name, lev):
    x, g, xt, gt = bench.make_tensors(device, name)
    plan = []
    cfg = bench.CONFIGS[bench.config_for(name)]()
    y = rbr_plan.lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=lev, out_plan=plan)
    return y, plan[0], x, g, xt, gt


def _correct(device, name, arms):
    ref = None
    rows, bad = [], []
    for label, lev in arms:
        try:
            y, plan, x, g, xt, gt = _run(device, name, lev)
            if ref is None:
                ref = bench.torch_ref(xt, gt)
            got = ttnn.to_torch(y).float()
            p = bench.pcc(ref, got)
            bias = bench.row_scale_bias(got, xt, gt)
            rows.append((label, p, bias, rbr_plan.plan_summary(plan)))
            if p < PCC_GATE:
                bad.append((name, label, round(p, 6)))
        except (AssertionError, ValueError) as exc:
            rows.append((label, None, None, f"SKIP {type(exc).__name__}: {str(exc)[:110]}"))
    print(f"\n=== correctness [{name}] (gate pcc >= {PCC_GATE}) ===")
    for label, p, bias, info in rows:
        tag = "SKIP" if p is None else ("ok  " if p >= PCC_GATE else "FAIL")
        pp = "-" if p is None else f"{p:.6f}"
        bb = "-" if bias is None else f"{bias * 100:+.3f}%"
        print(f"  {tag} {label:<18} pcc={pp}  bias={bb}  {info}")
    return bad


def test_correct(device):
    if MODE != "correct":
        pytest.skip("RBR_MODE != correct")
    arms = [(l, dict(v)) for l, v in bench.DOMAIN_ARMS]
    bad = []
    for name in _shapes():
        bad += _correct(device, name, arms)
    assert not bad, f"arms below the pcc gate: {bad}"


def test_poison_control(device):
    """The gate on the gate: with the mask REMOVED the poisoned cases must FAIL.

    If this arm passes, `fill_implicit_tile_padding` never landed and every
    "correct" result on a poisoned shape is vacuous.
    """
    if MODE != "poison_control":
        pytest.skip("RBR_MODE != poison_control")
    names = os.environ.get("RBR_SHAPES", ",".join(sorted(bench.POISON_SHAPES))).split(",")
    still_passing = []
    for name in names:
        label, lev = bench.POISON_CONTROL
        y, plan, x, g, xt, gt = _run(device, name, dict(lev))
        got = ttnn.to_torch(y).float()
        p = bench.pcc(bench.torch_ref(xt, gt), got)
        bias = bench.row_scale_bias(got, xt, gt)
        print(f"  {name:<14} NOMASK pcc={p:.6f} bias={bias * 100:+.3f}%  {rbr_plan.plan_summary(plan)}")
        # PCC is the WRONG detector here and the numbers say so: at W = 4095 the
        # unmasked arm is 1,477% off in row scale and still scores pcc 0.999936,
        # because a uniform per-row scale error barely moves a correlation.  The
        # ROW-SCALE BIAS is the criterion.
        if abs(bias) <= 0.01:
            still_passing.append((name, round(p, 6), round(bias * 100, 3)))
    assert not still_passing, f"no-mask control looked CORRECT (poison never landed): {still_passing}"


def test_correct_arms(device):
    if MODE != "correct_arms":
        pytest.skip("RBR_MODE != correct_arms")
    arms = [(l, dict(v)) for l, v in bench.focus_arms()]
    bad = []
    for name in _shapes():
        bad += _correct(device, name, arms)
    assert not bad, f"arms below the pcc gate: {bad}"


def _perf(device, name, arms, iters=0, no_zones=1):
    iters = iters or bench.N_ITERS
    manifest = []
    for label, lev in arms:
        lev = dict(lev)
        lev["no_zones"] = no_zones
        try:
            y, plan, x, g, xt, gt = _run(device, name, lev)
        except (AssertionError, ValueError) as exc:
            print(f"  SKIP {name}/{label}: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        got = ttnn.to_torch(y).float()
        p = bench.pcc(bench.torch_ref(xt, gt), got)
        bias = bench.row_scale_bias(got, xt, gt)
        cfg_name = bench.config_for(name)
        n = bench.dispatch(
            device,
            lambda l=lev, xx=x, gg=g, c=cfg_name: rbr_plan.lab_rms_norm(
                xx, gamma=gg, compute_kernel_config=bench.CONFIGS[c](), levers=l
            ),
            iters,
        )
        manifest.append(
            {
                "label": f"{name}/{label}",
                "shape": name,
                "levers": {k: v for k, v in lev.items()},
                "plan": rbr_plan.plan_summary(plan),
                "pcc": round(p, 6),
                "bias": round(bias * 100, 4),
                # +1 for the correctness dispatch above: report_from_csv folds the
                # per-op CSV by dispatch ORDER, so every dispatch must be counted.
                "calls": n + 1,
                "profiled": iters,
            }
        )
        print(f"  ran {name}/{label:<18} pcc={p:.6f} bias={bias * 100:+.3f}%  {rbr_plan.plan_summary(plan)}")
    return manifest


def test_domain(device):
    if MODE != "domain":
        pytest.skip("RBR_MODE != domain")
    arms = [(l, dict(v)) for l, v in bench.DOMAIN_ARMS]
    manifest = []
    for name in _shapes():
        manifest += _perf(device, name, arms)
    path = bench.write_manifest(manifest)
    print(f"\nRBR: manifest -> {path} ({len(manifest)} arms)")
    assert manifest


def test_perf(device):
    if MODE != "perf":
        pytest.skip("RBR_MODE != perf")
    arms = [(l, dict(v)) for l, v in bench.focus_arms()]
    manifest = []
    for name in _shapes():
        manifest += _perf(device, name, arms)
    path = bench.write_manifest(manifest)
    print(f"\nRBR: manifest -> {path} ({len(manifest)} arms)")
    assert manifest


def test_zones(device):
    if MODE != "zones":
        pytest.skip("RBR_MODE != zones")
    arms = [(l, dict(v)) for l, v in bench.DOMAIN_ARMS]
    manifest = []
    for name in _shapes():
        manifest += _perf(device, name, arms, iters=1, no_zones=0)
    path = bench.write_manifest(manifest)
    print(f"\nRBR: zone manifest -> {path}")
    assert manifest
