# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf experiment `tile_row_stealing` — validation of the GRADUATED form (graduate/, = graduation.diff applied).

The op module's select_plan / create_program_descriptor / rewrite_addresses are swapped for graduate/'s
(work stealing + its plan defaults: quartic body and split 16 on work-stealing plans). Each cell is compared
bit-for-bit with the unmodified op pinned to the same silu_body / split_rows (data movement is exact, so the
result depends only on the body), across a fresh build and program-cache hits.

    scripts/run_safe_pytest.sh --run-all <this file> -k grad_bitident
    GQKV_STEAL_PERF=1 ... -k "grad_perf"   (prints STEAL_NS grad-<case> <ns>)
"""

import importlib
import os

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    _check_case,
    _run,
)
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_reader_addressing import _run_and_check


op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
CASES = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.tile_row_stealing.test_steal"
).CASES
grad = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.tile_row_stealing.graduate."
    "gen_qkv_causal_conv1d_silu_program_descriptor"
)


def _invoke(device, case, limits):
    with op_module.plan_limits(**limits):
        if case["inputs"][3] is not None:
            return _run_and_check(device, case), None, None, op_module.get_last_plan()
        outputs, host, device_inputs = _run(device, case)
        return outputs, host, device_inputs, op_module.get_last_plan()


def _graduated(mp, stress=False):
    mp.setattr(op_module, "select_plan", grad.select_plan)
    mp.setattr(op_module, "create_program_descriptor", grad.create_program_descriptor)
    mp.setattr(op_module, "rewrite_addresses", grad.rewrite_addresses)
    mp.setattr(grad, "STEAL_STRESS", stress)


GRAD_CELLS = [
    ("conv1d-004", {}, False),
    ("conv1d-004", {}, True),
    ("conv1d-020", {}, False),
    ("conv1d-020", {}, True),
    ("conv1d-019", {}, False),
    ("conv1d-019", {}, True),
    ("conv1d-020", dict(max_block_rows=3), True),  # ragged blocks
    ("conv1d-012", dict(max_cores_per_group=2), True),
    ("conv1d-011", dict(max_cores_per_group=2), False),  # fp32 DEST multi-row: library body, split 24
    ("conv1d-001", {}, False),  # nothing to steal: static
    ("conv1d-003", {}, False),
    ("conv1d-005", {}, False),
    ("conv1d-010", {}, False),
]


@pytest.mark.parametrize(
    "case_id,limits,stress",
    GRAD_CELLS,
    ids=[
        f"{c}-{'_'.join(f'{k}{v}' for k, v in l.items()) or 'plan'}{'-stress' if s else ''}" for c, l, s in GRAD_CELLS
    ],
)
def test_grad_bitident(device, monkeypatch, case_id, limits, stress):
    import torch

    case = CASES[case_id]
    op_module._PROGRAM_CACHE.clear()
    with monkeypatch.context() as mp:
        _graduated(mp, stress)
        runs = [_invoke(device, case, limits) for _ in range(3)]  # build + 2 cache hits
    op_module._PROGRAM_CACHE.clear()
    plan = runs[0][3]
    pins = dict(limits, silu_body=plan.silu_body, split_rows=plan.split_rows)
    reference, host, device_inputs, plan0 = _invoke(device, case, pins)
    op_module._PROGRAM_CACHE.clear()
    if host is not None:
        _check_case(case, reference, host, device_inputs)
    ref = [ttnn.to_torch(t) for t in reference]
    for outputs, host, device_inputs, _ in runs:
        if host is not None:
            _check_case(case, outputs, host, device_inputs)
        for a, b in zip(ref, outputs):
            assert torch.equal(a, ttnn.to_torch(b)), f"{case_id} {limits}: graduated result differs"
    print(f"GRAD_PLAN {case_id} {limits}: {plan}")


@pytest.mark.skipif(os.environ.get("GQKV_STEAL_PERF") != "1", reason="perf bench: set GQKV_STEAL_PERF=1")
@pytest.mark.parametrize("case_id", ["conv1d-004", "conv1d-020", "conv1d-019"])
def test_grad_perf(device, monkeypatch, case_id):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    op_module._PROGRAM_CACHE.clear()
    with monkeypatch.context() as mp:
        _graduated(mp)
        outputs, host, device_inputs, plan = _invoke(device, CASES[case_id], {})
    op_module._PROGRAM_CACHE.clear()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    latest = ttnn.get_latest_programs_perf_data()
    durations = [
        res.duration
        for programs in latest.values()
        for program in programs
        for name, res in program.program_analyses_results.items()
        if name.startswith("DEVICE KERNEL DURATION")
    ]
    print(f"\nSTEAL_NS grad-{case_id} {' '.join(str(d) for d in durations)}")
    print(f"PLAN {case_id} grad: {plan}")
    if host is not None:
        _check_case(CASES[case_id], outputs, host, device_inputs)
