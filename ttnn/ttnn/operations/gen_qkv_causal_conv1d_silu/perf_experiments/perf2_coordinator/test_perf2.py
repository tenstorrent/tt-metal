# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 2 coordinator harness: whole-op device-ns of the real op under plan pins (measurement only).

    GQKV_PERF2=1 scripts/run_safe_pytest.sh --profile --no-py-zones --run-all \
        "<this file>::test_op[<variant>-<case_id>]" ...

One GenericOpDeviceOperation row per test, in the order given (op CSV, DEVICE KERNEL DURATION [ns]).
Variant "base" runs the op as of HEAD before Perf 2 (head/: descriptor + kernels frozen from git) for a
same-session before/after.
Every variant checks correctness against the float64 reference (the op's golden-style check), so a
pin that breaks the op fails loudly.

Cases: every golden cell the op's tests know by id, plus two cells the golden suite does not carry:
  020fp32   conv1d-020's shape with a 32-bit DEST: a multi-row work-stealing plan that must run the
            library SiLU body (fp32 accuracy); decides the split_rows rule for library-body stealing plans.
  u1multi   [1, 1024, 1952], (640, 640, 672): C_t = 61 is prime, so the plan is 61 groups x ONE core
            x 32 tile-rows — a multi-row plan with nothing to steal; decides the SiLU body there.
"""

import importlib
import os

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _case,
    _check_case,
    _run,
)
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_reader_addressing import (
    GOLDEN_CASES,
    _run_and_check,
)

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
# The op as of HEAD before Perf 2 (descriptor + kernels frozen in head/): the same-session baseline.
head = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.perf2_coordinator.head."
    "gen_qkv_causal_conv1d_silu_program_descriptor"
)

_RM = ttnn.ROW_MAJOR_LAYOUT
EXTRA = [
    _case("020fp32", ((1, 2048, 2048), (1024, 512, 512), 1024, None, 0), _RM, fp32=True),
    _case("u1multi", ((1, 1024, 1952), (640, 640, 672), 32, None, 0), _RM),
]
CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES + EXTRA}

LIB, BF16 = 0, 1
VARIANTS = {
    "default": {},
    "s8": dict(split_rows=8),
    "s16": dict(split_rows=16),
    "s24": dict(split_rows=24),
    "lib": dict(silu_body=LIB),
    "lib_s16": dict(silu_body=LIB, split_rows=16),
    "lib_s24": dict(silu_body=LIB, split_rows=24),
    "bf16": dict(silu_body=BF16),
    "bf16_s24": dict(silu_body=BF16, split_rows=24),
    "base": None,  # the pre-Perf-2 op (head/), its own defaults
}


@pytest.mark.skipif(os.environ.get("GQKV_PERF2") != "1", reason="perf harness: set GQKV_PERF2=1")
@pytest.mark.parametrize("case_id", sorted(CASES))
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_op(device, monkeypatch, case_id, variant):
    case = CASES[case_id]
    op_module._PROGRAM_CACHE.clear()
    limits = VARIANTS[variant]
    if limits is None:
        for name in ("select_plan", "create_program_descriptor", "rewrite_addresses"):
            monkeypatch.setattr(op_module, name, getattr(head, name))
        limits = {}
    with op_module.plan_limits(**limits):
        if case["inputs"][3] is not None:
            _run_and_check(device, case)
        else:
            outputs, host, device_inputs = _run(device, case)
            _check_case(case, outputs, host, device_inputs)
        print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
    op_module._PROGRAM_CACHE.clear()
