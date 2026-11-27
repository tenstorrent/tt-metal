# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 2: work stealing (Plan.work_stealing) — tile-row blocks claimed at run time inside a channel group.

Data movement is exact and the tap sum order is fixed, so a cell's result depends only on its SiLU body:
every stealing run must be bit-identical (torch.equal) to the same cell on the static path. The static
reference pins max_cores_per_group=1 (one core per channel group: nothing to steal) and the same silu_body.
Each stealing cell runs a fresh build plus two program-cache hits (the claim semaphores must reset every
launch), with the golden-style accuracy check on every run, in the default schedule and in the stress
schedule (STEAL_STRESS: every core steals before its own range — every claim race, every DRAM halo).

Cells: the golden multi-row plans (004, 020, 008 taps in L1, 019 TILE + fp32 DEST + L1 + channel-paged),
plus pinned plans for ragged blocks (020 block_rows 3), channel-paged history (012), fp32 DEST (011), and a
256-token RM / TILE pair pinned to a narrow-group stealing plan (w 2, 2 cores x 4 tile-rows per group).
"""

import importlib

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

# ARCHIVED (Perf 2): this suite targets the work-stealing op in ../stealing_plus_silu_attempt.diff, which did NOT
# graduate — its NoC fetch-and-add claim hung intermittently on device (see changelog Perf 2). Apply that diff
# (and drop this marker) to run it.
pytestmark = pytest.mark.skip(reason="archived: needs perf2_coordinator/stealing_plus_silu_attempt.diff applied")

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
desc = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)

CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES}
CASES["rmT256"] = _case("rmT256", ((1, 256, 2560), (512, 512, 1536), 256, None, 0), ttnn.ROW_MAJOR_LAYOUT).values[0]
CASES["tileT256"] = _case("tileT256", ((1, 256, 2560), (512, 512, 1536), 256, None, 0), ttnn.TILE_LAYOUT).values[0]


def _invoke(device, case, limits):
    with op_module.plan_limits(**limits):
        if case["inputs"][3] is not None:
            outputs = _run_and_check(device, case)
            return outputs, op_module.get_last_plan()
        outputs, host, device_inputs = _run(device, case)
        _check_case(case, outputs, host, device_inputs)
        return outputs, op_module.get_last_plan()


CELLS = [
    ("conv1d-004", {}),
    ("conv1d-020", {}),
    ("conv1d-008", {}),
    ("conv1d-019", {}),
    ("conv1d-020", dict(max_block_rows=3)),  # ragged blocks mid-sequence
    ("conv1d-012", dict(max_cores_per_group=2)),  # channel-paged history halo on stolen blocks
    ("conv1d-011", dict(max_cores_per_group=2)),  # 32-bit DEST: library body, split 24
    ("rmT256", dict(max_cores_per_group=3)),  # pinned: w 2, 40 groups x 2 cores x 4 tile-rows
    ("tileT256", dict(max_cores_per_group=3)),  # TILE: the extra untilized halo row on stolen blocks
]


@pytest.mark.parametrize("stress", [False, True], ids=["schedule", "stress"])
@pytest.mark.parametrize(
    "case_id,limits", CELLS, ids=[f"{c}-{'_'.join(f'{k}{v}' for k, v in l.items()) or 'plan'}" for c, l in CELLS]
)
def test_stealing_bit_identical(device, monkeypatch, case_id, limits, stress):
    import torch

    case = CASES[case_id]
    op_module._PROGRAM_CACHE.clear()
    with monkeypatch.context() as mp:
        mp.setattr(desc, "STEAL_STRESS", stress)
        runs = [_invoke(device, case, limits) for _ in range(3)]  # build + 2 program-cache hits
    op_module._PROGRAM_CACHE.clear()
    plan = runs[0][1]
    assert plan.work_stealing, f"{case_id} {limits}: expected a work-stealing plan, got {plan}"
    static_limits = dict(limits, max_cores_per_group=1, silu_body=plan.silu_body)
    reference, static_plan = _invoke(device, case, static_limits)
    op_module._PROGRAM_CACHE.clear()
    assert not static_plan.work_stealing
    ref = [ttnn.to_torch(t) for t in reference]
    for outputs, _ in runs:
        for name, a, b in zip("qkv", ref, outputs):
            assert torch.equal(a, ttnn.to_torch(b)), f"{case_id} {limits} stress={stress}: {name} differs from static"


@pytest.mark.parametrize(
    "seq_tiles, ch_tiles, fp32, stealing, split",
    [
        (1, 80, False, False, desc.SPLIT_READER_ROWS),  # conv1d-001: one block per core
        (4, 80, False, False, desc.SPLIT_READER_ROWS),  # conv1d-003
        (128, 80, False, True, desc.SPLIT_READER_ROWS_STEALING),  # conv1d-004
        (64, 64, False, True, desc.SPLIT_READER_ROWS_STEALING),  # conv1d-020
        (64, 64, True, True, desc.SPLIT_READER_ROWS),  # 32-bit DEST (library body): measured carve-out
        (32, 61, False, False, desc.SPLIT_READER_ROWS),  # one core per channel group: nothing to steal
    ],
)
def test_stealing_plan_rule(seq_tiles, ch_tiles, fp32, stealing, split):
    plan = desc.select_plan(
        seq_tiles,
        ch_tiles,
        11,
        10,
        desc.REGIME_RM,
        dest_tiles=desc.dest_capacity_tiles(fp32, False),
        fp32_dest_acc_en=fp32,
    )
    blocks = -(-plan.core_seq_tiles_max // plan.block_rows)
    assert plan.work_stealing == (blocks > 1 and plan.cores_per_group > 1) == stealing
    assert plan.split_rows == split
