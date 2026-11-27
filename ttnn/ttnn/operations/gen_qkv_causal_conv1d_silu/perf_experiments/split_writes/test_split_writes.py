# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT split_writes: share the q / k / v DRAM writes between the writer (BRISC / NoC1) and
the reader (NCRISC / NoC0) on the op's full grid plan.

  test_bitident[variant-case]  baseline op (unmodified) and the variant on the same inputs:
                               torch.equal on q / k / v + the op's own accuracy check on the variant.
  test_perf[variant-case]      ONE op call (one GenericOpDeviceOperation row under --profile).

Variant = (NCRISC share S of every P tiles, poll the read barrier, plan_limits pins). "base" is the
unmodified op; "sw0" is this dir's kernels with S = 0 (control: the baseline sequence + the extra CB).

    scripts/run_safe_pytest.sh --no-jit-server --profile --no-py-zones \
        "<this file>::test_perf[base-conv1d-004]" "<this file>::test_perf[s6-conv1d-004]"
    python3 <this dir>/rows.py      # device kernel ns per row of the newest ops CSV
"""

import importlib

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
)
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_reader_addressing import (
    GOLDEN_CASES,
    _run_and_check,
)

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
desc_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)
grad_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.split_writes.graduate.gen_qkv_causal_conv1d_silu_program_descriptor"
)
sw_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.split_writes.split_writes_descriptor"
)

pytestmark = pytest.mark.use_module_device  # one device open per module (the op's test dir does the same)

CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES}
Q = dict(silu_body=desc_module.SILU_BF16_QUARTIC)

G = sw_module.gradient_share
# name -> (share, period, poll_barrier, plan_limits[, dynamic[, share_fn]]) ; share None = the unmodified op
VARIANTS = {
    # the clean graduation candidate (graduate/: the op's descriptor + kernels with WRITE_SHARE = 4)
    "grad": ("graduate", 8, True, {}),
    "grad_q": ("graduate", 8, True, Q),
    "g26": (0, 8, True, {}, False, G(2, 6)),
    "g26_q": (0, 8, True, Q, False, G(2, 6)),
    "g36_q": (0, 8, True, Q, False, G(3, 6)),
    "g17_q": (0, 8, True, Q, False, G(1, 7)),
    "g35_q": (0, 8, True, Q, False, G(3, 5)),
    "dyn": (0, 8, True, {}, True),
    "dyn_q": (0, 8, True, Q, True),
    "dynnp_q": (0, 8, False, Q, True),
    "ovf": (0, 8, True, {}, 2),
    "ovf_q": (0, 8, True, Q, 2),
    "base": (None, 8, True, {}),
    "sw0": (0, 8, True, {}),
    "s2": (2, 8, True, {}),
    "s4": (4, 8, True, {}),
    "s6": (6, 8, True, {}),
    "s8": (8, 8, True, {}),
    "s4np": (4, 8, False, {}),
    "s6np": (6, 8, False, {}),
    # split-reader rebalanced with the writes moved
    "s8sp32": (8, 8, True, dict(split_rows=32)),
    "s4sp32": (4, 8, True, dict(split_rows=32)),
    "s6sp32": (6, 8, True, dict(split_rows=32)),
    "base_sp32": (None, 8, True, dict(split_rows=32)),
    # quartic SiLU body pinned (multi-row plans default to the library body)
    "base_q": (None, 8, True, Q),
    "s2_q": (2, 8, True, Q),
    "s3_q": (3, 8, True, Q),
    "s4_q": (4, 8, True, Q),
    "s5_q": (5, 8, True, Q),
    # quartic + split-reader pins (rebalance the reads with the writes)
    "base_sp16_q": (None, 8, True, dict(Q, split_rows=16)),
    "s4sp16_q": (4, 8, True, dict(Q, split_rows=16)),
    "s4sp20_q": (4, 8, True, dict(Q, split_rows=20)),
    "s4sp12_q": (4, 8, True, dict(Q, split_rows=12)),
    "s3sp16_q": (3, 8, True, dict(Q, split_rows=16)),
    "s5sp16_q": (5, 8, True, dict(Q, split_rows=16)),
    "s2sp8_q": (2, 8, True, dict(Q, split_rows=8)),
    "s6_q": (6, 8, True, Q),
    "s8_q": (8, 8, True, Q),
}

CASE_IDS = [
    "conv1d-004",
    "conv1d-001",
    "conv1d-002",
    "conv1d-003",
    "conv1d-020",
    "conv1d-005",
    "conv1d-006",
    "conv1d-010",
    "conv1d-011",
    "conv1d-019",
    "conv1d-012",
    "conv1d-013",
    "conv1d-014",
    "conv1d-016",
]


# Writes-only ablation (mechanism check, wrong results by construction — test_ablation only): compute
# reduced to its CB sequence, no input / history / halo reads (test_perf_t1.py's defines).
ABLATE_WRITES_ONLY = {
    2: [("GQKV_ABLATE_SILU", "1"), ("GQKV_ABLATE_MAC", "1"), ("GQKV_ABLATE_TILIZE", "1")],
    desc_module.READER_KERNEL_INDEX: [("GQKV_ABLATE_READS", "1")],
    desc_module.WRITER_KERNEL_INDEX: [("GQKV_ABLATE_READS", "1")],
}


def _install(monkeypatch, variant, extra_defines=None):
    share, period, poll, limits = VARIANTS[variant][:4]
    dynamic = len(VARIANTS[variant]) > 4 and VARIANTS[variant][4]
    share_fn = VARIANTS[variant][5] if len(VARIANTS[variant]) > 5 else None
    op_module._PROGRAM_CACHE.clear()
    if share is None:
        create = desc_module.create_program_descriptor
    elif share == "graduate":
        create = grad_module.create_program_descriptor
    else:
        create = sw_module.make_create(share, period, poll, dynamic, share_fn)
    if extra_defines:
        inner = create

        def create(*args, **kwargs):
            descriptor, work = inner(*args, **kwargs)
            kernels = list(descriptor.kernels)
            for index, defs in extra_defines.items():
                kernels[index].defines = list(kernels[index].defines) + list(defs)
            descriptor.kernels = kernels
            return descriptor, work

    monkeypatch.setattr(op_module, "create_program_descriptor", create)
    return limits


def _call(device, case):
    """One op call; returns the outputs (channel-paged cells go through their own checked runner)."""
    if case["inputs"][3] is not None:
        return _run_and_check(device, case), None, None
    return _run(device, case)


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "base"])
def test_bitident(device, monkeypatch, case_id, variant):
    import torch  # local: no global torch import under ttnn/ (pre-commit)

    case = CASES[case_id]
    limits = _install(monkeypatch, variant)
    with op_module.plan_limits(**limits):
        outputs, host, device_inputs = _call(device, case)
    plan = op_module.get_last_plan()
    if host is not None:
        _check_case(case, outputs, host, device_inputs)
    # baseline: the unmodified op, same plan pins, same (seeded) inputs
    _install(monkeypatch, "base")
    with op_module.plan_limits(**limits):
        reference, _, _ = _call(device, case)
    print(f"PLAN {case_id} {variant}: {plan}")
    op_module._PROGRAM_CACHE.clear()
    for name, a, b in zip("qkv", reference, outputs):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"{variant} {case_id}: {name} differs from the baseline"


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_perf(device, monkeypatch, case_id, variant):
    case = CASES[case_id]
    limits = _install(monkeypatch, variant)
    with op_module.plan_limits(**limits):
        _call(device, case)
    print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
    op_module._PROGRAM_CACHE.clear()


@pytest.mark.parametrize("case_id", ["conv1d-004", "conv1d-020"])
@pytest.mark.parametrize("variant", ["base", "s2", "s4", "s6", "ovf", "g26"])
def test_ablation_writes_only(device, monkeypatch, case_id, variant):
    case = CASES[case_id]
    limits = _install(monkeypatch, variant, ABLATE_WRITES_ONLY)
    with op_module.plan_limits(**limits):
        _call(device, case)
    op_module._PROGRAM_CACHE.clear()


@pytest.mark.parametrize("case_id", ["conv1d-003", "conv1d-004", "conv1d-005"])
def test_grad_cache_hit(device, monkeypatch, case_id):
    import torch  # local: no global torch import under ttnn/ (pre-commit)

    """Graduation candidate on a program-cache hit: the second call reuses the descriptor and only
    rewrites buffer addresses (incl. the reader's new q / k / v slots) — must match the baseline."""
    case = CASES[case_id]
    _install(monkeypatch, "base")
    reference, _, _ = _call(device, case)
    _install(monkeypatch, "grad")
    monkeypatch.setattr(op_module, "rewrite_addresses", grad_module.rewrite_addresses)
    first, _, _ = _call(device, case)
    keep = [ttnn.to_torch(t) for t in first]
    del first
    _ = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 32, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)  # shift
    second, host, device_inputs = _call(device, case)  # cache hit
    assert len(op_module._PROGRAM_CACHE) == 1
    _check_case(case, second, host, device_inputs)
    op_module._PROGRAM_CACHE.clear()
    for name, a, b, c in zip("qkv", reference, second, keep):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"cache hit: {name} differs from the baseline"
        assert torch.equal(ttnn.to_torch(a), c), f"first call: {name} differs from the baseline"
