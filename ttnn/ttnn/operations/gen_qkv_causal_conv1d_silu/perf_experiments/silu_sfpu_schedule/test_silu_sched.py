# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf experiment `silu_sfpu_schedule` — a cheaper 16-bit-DEST SiLU body at the same precision.

The real op runs with its compute kernel swapped for kernels/compute_silu_sched.cpp (a copy of the op's
compute kernel, permanent stage zones kept) and the SiLU body pinned by the define GQKV_SILU_BODY:
    0 library calculate_silu         (baseline a: the multi-row default)
    1 sfpi quartic silu_bf16_rows    (baseline b: the one-row default)
    2 raw-TTI quartic, const-LReg clamps, no load macros   (same arithmetic as 1)
    3 raw-TTI quartic + SFPLOADMACRO BEG / FIN              (same arithmetic as 1)

Accuracy (isolated SiLU: taps 0,0,0,1, so y = x exactly in a 16-bit DEST):
    scripts/run_safe_pytest.sh --no-jit-server --run-all <this file> -k accuracy -s
Perf (one op call per test; DEVICE KERNEL DURATION read in-process, printed "SCHED_NS <id> <ns>"):
    run_perf.sh <variant>-<case> ...
"""

import importlib
import os
from pathlib import Path

import pytest
import ttnn

from models.common.utility_functions import comp_pcc
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
    pytorch_reference,
)
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_reader_addressing import (
    GOLDEN_CASES,
    _run_and_check,
)

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
desc_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)

KERNEL = Path(__file__).parent / "kernels" / "compute_silu_sched.cpp"
READER, WRITER, COMPUTE = desc_module.READER_KERNEL_INDEX, desc_module.WRITER_KERNEL_INDEX, 2
CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES}
BODIES = {"lib": 0, "q": 1, "raw": 2, "mac": 3, "mac4": 4, "f16": 5, "f16g": 6, "cub": 7}
TAP_COUNT, HISTORY_ROWS = 4, 3


GRAD_KERNEL = Path(__file__).parent / "graduate" / "gen_qkv_causal_conv1d_silu_compute.cpp"
GRAD = "grad"  # body token: the proposed op kernel (graduate/), body chosen by the plan (CT arg 4)


def _patch(monkeypatch, body, compute_defines=(), reads=False, writes=False):
    """Swap in the experiment compute kernel with GQKV_SILU_BODY = body (None: the op's own kernel;
    GRAD: the proposed graduation kernel, body from the plan)."""
    real = op_module.create_program_descriptor

    def create(*args, **kwargs):
        descriptor, work = real(*args, **kwargs)
        kernels = list(descriptor.kernels)
        if body == GRAD:
            kernels[COMPUTE].kernel_source = str(GRAD_KERNEL)
            kernels[COMPUTE].defines = list(compute_defines)
        elif body is not None:
            kernels[COMPUTE].kernel_source = str(KERNEL)
            kernels[COMPUTE].defines = [("GQKV_SILU_BODY", str(body))] + list(compute_defines)
        if reads:
            kernels[READER].defines = [("GQKV_ABLATE_READS", "1")]
            kernels[WRITER].defines = [("GQKV_ABLATE_READS", "1")]
        if writes:
            kernels[WRITER].defines = list(kernels[WRITER].defines) + [("GQKV_ABLATE_WRITES", "1")]
        descriptor.kernels = kernels
        return descriptor, work

    monkeypatch.setattr(op_module, "create_program_descriptor", create)


# ---------------------------------------------------------------------------------------------
# Accuracy: the SiLU in isolation (same construction as tests/.../test_silu_body.py)
# ---------------------------------------------------------------------------------------------


def _bf16_ordered(t):
    import torch

    bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int64)
    return torch.where(bits < 0, -(bits + (1 << 15)), bits)


def _all_finite_bf16(n):
    """Every finite bf16 value (both signs, normals, zeros, denormals), repeated to n elements."""
    import torch

    bits = torch.arange(0, 1 << 16, dtype=torch.int64)
    vals = bits.to(torch.int16).view(torch.bfloat16)
    vals = vals[torch.isfinite(vals.float())]
    reps = (n + vals.numel() - 1) // vals.numel()
    return vals.repeat(reps)[:n]


def _operands(seq, channels, kind):
    import torch

    if kind == "randn":
        gen = torch.Generator().manual_seed(1234)
        x = torch.randn(1, seq, channels, generator=gen) * torch.tensor([0.5, 2.0, 6.0, 20.0]).repeat(channels // 4)
        extremes = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 87.0, 88.5, 90.0, 200.0, -87.0, -88.5, -90.0, -200.0])
        x[0, 0, : extremes.numel()] = extremes
        x = x.to(torch.bfloat16)
    else:
        x = _all_finite_bf16(seq * channels).reshape(1, seq, channels)
    history = torch.zeros(1, HISTORY_ROWS, channels, dtype=torch.bfloat16)
    taps = [torch.zeros(1, 1, channels, dtype=torch.bfloat16) for _ in range(TAP_COUNT)]
    taps[-1] = torch.ones(1, 1, channels, dtype=torch.bfloat16)
    return x, history, taps


def _run_silu(device, x, history, taps, widths):
    import torch

    to_dev = lambda t, layout: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device)
    history_tt = to_dev(history, ttnn.ROW_MAJOR_LAYOUT)
    outputs = op_module.gen_qkv_causal_conv1d_silu(
        to_dev(x, ttnn.ROW_MAJOR_LAYOUT),
        history_tt,
        *[to_dev(t, ttnn.TILE_LAYOUT) for t in taps],
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
        actual_start=ttnn.from_torch(
            torch.tensor([0], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        ),
        predecessor_carry=history_tt,
    )
    seq = x.shape[1]
    return torch.cat([ttnn.to_torch(o).reshape(1, seq, -1) for o in outputs], dim=-1)


def _metrics(got, x):
    import torch

    xd = x.double()
    exact = (xd * torch.sigmoid(xd)).to(torch.bfloat16)
    ulp = (_bf16_ordered(got) - _bf16_ordered(exact)).abs()
    # ULP statistics where the result is a normal bf16 number: x >= -80 (below, both shipped bodies flush the
    # exp overflow to silu = 0; true |silu| < 1e-33) and |x| >= 2^-120 (silu(x) ~ x/2 stays normal).
    sel = (xd >= -80.0) & ((xd.abs() >= 2.0**-120) | (xd == 0))
    u = ulp[sel]
    err = got.double()[sel] - (xd * torch.sigmoid(xd))[sel]
    ref = (xd * torch.sigmoid(xd))[sel]
    rel_rms = float(err.pow(2).mean().sqrt() / ref.std())
    big_ok = bool(torch.equal(got[xd >= 90.0], x[xd >= 90.0]))
    neg_ok = bool((got[xd <= -90.0].float().abs() <= 1e-30).all())
    return dict(
        exact=float((u == 0).float().mean()),
        max_ulp=int(u.max()),
        n_ulp1=int((u == 1).sum()),
        n=int(u.numel()),
        rel_rms=rel_rms,
        big_ok=big_ok,
        neg_ok=neg_ok,
        finite=bool(torch.isfinite(got.float()).all()),
    )


@pytest.mark.parametrize("kind", ["randn", "allbf16"])
@pytest.mark.parametrize(
    "shape, widths",
    [((1, 32, 2560), (512, 512, 1536)), ((1, 2048, 2048), (1024, 512, 512))],
    ids=["conv1d-001_one_row", "conv1d-020_multi_row"],
)
def test_accuracy(device, monkeypatch, shape, widths, kind):
    import torch

    _, seq, channels = shape
    x, history, taps = _operands(seq, channels, kind)
    got = {}
    for name, body in BODIES.items():
        op_module._PROGRAM_CACHE.clear()
        with monkeypatch.context() as mp:
            _patch(mp, body)
            got[name] = _run_silu(device, x, history, taps, widths)
        op_module._PROGRAM_CACHE.clear()
        m = _metrics(got[name], x)
        print(
            f"\nACC {shape[1]}x{channels} {kind} body={name}: correctly_rounded={m['exact']:.5f} "
            f"max_ulp={m['max_ulp']} n_ulp1={m['n_ulp1']}/{m['n']} rel_rms={m['rel_rms']:.3e} "
            f"x>=90->x {m['big_ok']} x<=-90->0 {m['neg_ok']} finite {m['finite']}"
        )
        assert m["finite"] and m["big_ok"] and m["neg_ok"], f"{name}: clamp extremes"
    # raw / mac / mac4 run the sfpi quartic's arithmetic; f16g runs f16's (fp16 refit) arithmetic
    for name, ref in (("raw", "q"), ("mac", "q"), ("mac4", "q"), ("f16g", "f16")):
        same = got[name].view(torch.int16) == got[ref].view(torch.int16)
        print(f"BITIDENT {name} vs {ref}: {float(same.float().mean()):.6f} ({int((~same).sum())} differ)")
        assert bool(same.all()), f"{name} differs from {ref} in {int((~same).sum())} elements"
    diff = (_bf16_ordered(got["f16"]) - _bf16_ordered(got["q"])).abs()
    print(f"F16_VS_Q: {int((diff > 0).sum())} of {diff.numel()} elements differ, max {int(diff.max())} ULP")


# ---------------------------------------------------------------------------------------------
# Golden-style precision on whole cells (as test_gen_qkv_causal_conv1d_silu_precision_baseline.py)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body", list(BODIES))
@pytest.mark.parametrize("case_id", ["conv1d-004", "conv1d-003", "conv1d-020", "conv1d-001"])
def test_golden_precision(device, monkeypatch, case_id, body):
    import torch

    case = CASES[case_id]
    op_module._PROGRAM_CACHE.clear()
    _patch(monkeypatch, BODIES[body])
    outputs, host, device_inputs = _run(device, case)
    op_module._PROGRAM_CACHE.clear()
    _check_case(case, outputs, host, device_inputs)
    (_, seq, _), widths, _, _, _ = case["inputs"]
    _, conv_input, history, taps = host
    refs, _ = pytorch_reference(conv_input, history, taps, widths)
    actual = torch.cat([ttnn.to_torch(o).reshape(1, seq, -1).double() for o in outputs], dim=-1)
    expected = torch.cat(refs, dim=-1)
    _, pcc = comp_pcc(expected, actual)
    rel_rms = float((actual - expected).pow(2).mean().sqrt() / expected.std())
    ulp = (_bf16_ordered(actual) - _bf16_ordered(expected)).abs().float().flatten()
    p99 = float(ulp.kthvalue(max(1, int(0.99 * ulp.numel()))).values)
    print(f"\nGOLDEN {case_id} body={body}: pcc={pcc:.6f} rel_rms={rel_rms:.5f} ulp_p99={p99:.0f}")


# ---------------------------------------------------------------------------------------------
# Perf: one op call per test, device kernel ns read in-process
# ---------------------------------------------------------------------------------------------


def _variant(name):
    """<body>-<mode>[-rep<k>]: body in BODIES or 'op' (the unmodified op), mode full | nodm | nosilu."""
    parts = name.split("-")
    body = {"op": None, "grad": GRAD, "gradq": GRAD}.get(parts[0], BODIES.get(parts[0]))
    mode = parts[1]
    defines = []
    for p in parts[2:]:
        if p.startswith("rep"):
            defines.append(("GQKV_SILU_REPEAT", p[3:]))
        if p == "nostall":
            defines.append(("GQKV_TILE_STALL_OFF", "1"))
    if mode == "nosilu" or mode == "nodmnosilu":
        defines.append(("GQKV_ABLATE_SILU", "1"))
    nodm = mode in ("nodm", "nodmnosilu")
    check = mode == "full" and not any(d[0] == "GQKV_SILU_REPEAT" for d in defines)
    return body, defines, nodm, check


@pytest.mark.skipif(os.environ.get("GQKV_SCHED_PERF") != "1", reason="perf harness: set GQKV_SCHED_PERF=1")
@pytest.mark.parametrize(
    "case_id",
    [
        "conv1d-004",
        "conv1d-020",
        "conv1d-001",
        "conv1d-002",
        "conv1d-003",
        "conv1d-006",
        "conv1d-010",
        "conv1d-011",
        "conv1d-012",
        "conv1d-013",
        "conv1d-014",
        "conv1d-016",
        "conv1d-019",
    ],
)
@pytest.mark.parametrize(
    "variant", [f"{b}-{m}" for b in list(BODIES) + ["op", "grad", "gradq"] for m in ("full", "nodm")]
)
def test_perf(device, monkeypatch, case_id, variant):
    _perf(device, monkeypatch, case_id, variant)


def _perf(device, monkeypatch, case_id, variant):
    body, defines, nodm, check = _variant(variant)
    limits = dict(silu_body=desc_module.SILU_BF16_QUARTIC) if variant.startswith("gradq") else {}
    case = CASES[case_id]
    op_module._PROGRAM_CACHE.clear()
    if body is not None or defines or nodm:
        _patch(monkeypatch, 0 if body is None else body, defines, reads=nodm, writes=nodm)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # drop the operand uploads
    with op_module.plan_limits(**limits):
        if case["inputs"][3] is not None:
            _run_and_check(device, case)
            outputs = None
        else:
            outputs, host, device_inputs = _run(device, case)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    latest = ttnn.get_latest_programs_perf_data()
    durations = {
        name: res.duration
        for programs in latest.values()
        for program in programs
        for name, res in program.program_analyses_results.items()
        if name.startswith("DEVICE KERNEL DURATION")
    }
    kernel_ns = durations.get("DEVICE KERNEL DURATION [ns]")
    print(f"\nSCHED_NS {variant}-{case_id} {kernel_ns} {durations}")
    print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
    op_module._PROGRAM_CACHE.clear()
    if check and outputs is not None:
        _check_case(case, outputs, host, device_inputs)


@pytest.mark.skipif(os.environ.get("GQKV_SCHED_PERF") != "1", reason="perf harness: set GQKV_SCHED_PERF=1")
@pytest.mark.parametrize("case_id", ["conv1d-004", "conv1d-020", "conv1d-003", "conv1d-016", "conv1d-012"])
@pytest.mark.parametrize(
    "variant",
    [f"{b}-{m}-rep{k}" for b in BODIES for m in ("full", "nodm") for k in (1, 2, 3)]
    + [f"{b}-{m}-nostall" for b in BODIES for m in ("full", "nodm")]
    + ["lib-nodmnosilu"],
)
def test_pass(device, monkeypatch, case_id, variant):
    _perf(device, monkeypatch, case_id, variant)


# ---------------------------------------------------------------------------------------------
# The proposed op kernel (graduate/): its bf16 body is bit-identical to body 6 (f16g); multi-row plans keep the
# library body unless pinned.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["randn", "allbf16"])
@pytest.mark.parametrize(
    "shape, widths",
    [((1, 32, 2560), (512, 512, 1536)), ((1, 2048, 2048), (1024, 512, 512))],
    ids=["conv1d-001_one_row", "conv1d-020_multi_row"],
)
def test_grad_bitident(device, monkeypatch, shape, widths, kind):
    import torch

    _, seq, channels = shape
    x, history, taps = _operands(seq, channels, kind)
    got = {}
    for name, body, limits in (
        ("f16g", BODIES["f16g"], {}),
        ("lib", BODIES["lib"], {}),
        ("grad_pinned", GRAD, dict(silu_body=desc_module.SILU_BF16_QUARTIC)),
        ("grad_default", GRAD, {}),
    ):
        op_module._PROGRAM_CACHE.clear()
        with monkeypatch.context() as mp:
            _patch(mp, body)
            with op_module.plan_limits(**limits):
                got[name] = _run_silu(device, x, history, taps, widths)
                plan = op_module.get_last_plan()
        op_module._PROGRAM_CACHE.clear()
        got[name + "_plan"] = plan.silu_body
    default_ref = "f16g" if got["grad_default_plan"] == desc_module.SILU_BF16_QUARTIC else "lib"
    for name, ref in (("grad_pinned", "f16g"), ("grad_default", default_ref)):
        same = got[name].view(torch.int16) == got[ref].view(torch.int16)
        print(f"GRAD_BITIDENT {shape[1]}x{channels} {kind} {name} vs {ref}: {int((~same).sum())} differ")
        assert bool(same.all())
