# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf experiment `tile_row_stealing` — bit-identity gate + device-time bench (measurement only).

baseline  = the unmodified op (static contiguous token range per core)
candidate = the same op with create_program_descriptor swapped for steal_program_descriptor's
            (cross-core tile-row work stealing inside a channel group; see kernels/steal_reader.cpp)

Correctness:  scripts/run_safe_pytest.sh --run-all <this file> -k bitident
Perf (one op call per test; DEVICE KERNEL DURATION read in-process with ttnn.ReadDeviceProfiler, printed as
"STEAL_NS <variant>-<case> <ns>"; the device log with the stage zones lands in $TT_METAL_PROFILER_DIR/.logs):
    run_perf.sh <variant>-<case> ...
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
desc_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)
steal_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.tile_row_stealing.steal_program_descriptor"
)

CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES}
CASES["rmT256"] = _case("rmT256", ((1, 256, 2560), (512, 512, 1536), 256, None, 0), ttnn.ROW_MAJOR_LAYOUT).values[0]
CASES["tileT256"] = _case("tileT256", ((1, 256, 2560), (512, 512, 1536), 256, None, 0), ttnn.TILE_LAYOUT).values[0]
QUARTIC = desc_module.SILU_BF16_QUARTIC

MODES = {"auto": None, "static": 0, "own": 1, "on": 2, "eager": 3}


def _call(device, monkeypatch, case, limits, mode):
    """One op call. mode None -> the unmodified op; a name -> the stealing descriptor with that STEAL_MODE."""
    op_module._PROGRAM_CACHE.clear()
    with monkeypatch.context() as mp:
        if mode is not None:
            mp.setattr(steal_module, "STEAL_MODE", MODES[mode])
            mp.setattr(op_module, "create_program_descriptor", steal_module.create_program_descriptor)
        with op_module.plan_limits(**limits):
            if case["inputs"][3] is not None:  # channel-paged history: own operand builder (+ checks)
                outputs, host, device_inputs = _run_and_check(device, case), None, None
            else:
                outputs, host, device_inputs = _run(device, case)
            plan = op_module.get_last_plan()
    op_module._PROGRAM_CACHE.clear()
    return outputs, host, device_inputs, plan


# (case, plan_limits, modes) — the flagged multi-row cells, every regime / placement path, and pinned plans
# that put ragged (short) blocks and DRAM halos in the middle of a core's stolen sequence.
BITIDENT_CELLS = [
    ("conv1d-004", {}, ["auto", "own", "eager"]),
    ("conv1d-004", dict(silu_body=QUARTIC), ["auto", "eager"]),
    ("conv1d-020", {}, ["auto", "eager"]),
    ("conv1d-019", {}, ["auto", "eager"]),  # TILE, L1 in / out, channel-paged, fp32 DEST, 2 rows / core
    ("conv1d-003", {}, ["on", "eager"]),  # one row per core: protocol forced on
    ("conv1d-003", dict(max_cores_per_group=2), ["auto", "eager"]),  # 2 rows / core, DRAM halo at t0 > 0
    ("conv1d-011", dict(max_cores_per_group=2), ["auto", "eager"]),  # fp32 DEST, multi-row
    ("conv1d-012", dict(max_cores_per_group=2), ["auto", "eager"]),  # channel-paged history, multi-row
    ("conv1d-006", dict(max_cores_per_group=2), ["auto", "eager"]),  # L1 input, multi-row
    ("conv1d-010", dict(max_cores_per_group=2), ["auto", "eager"]),  # L1 output, multi-row
    ("conv1d-005", {}, ["on", "eager"]),  # TILE one row per core
    ("rmT256", dict(max_block_rows=3, max_cores_per_group=3), ["auto", "eager"]),  # ranges 3,3,2: short block
    ("rmT256", dict(max_block_rows=2, max_cores_per_group=3), ["auto", "eager"]),  # short blocks mid-range
    ("tileT256", dict(max_block_rows=2, max_cores_per_group=3), ["auto", "eager"]),  # TILE + short blocks
    ("tileT256", dict(max_cores_per_group=4), ["auto", "eager"]),  # TILE, 2 one-row blocks per core
    ("conv1d-001", {}, ["auto"]),  # single-row plan: static by construction
    ("conv1d-020", dict(max_block_rows=3), ["auto", "eager"]),  # 5/4-row ranges -> blocks 3+2 / 3+1 (ragged)
    ("conv1d-020", dict(max_block_rows=2), ["auto", "eager"]),  # blocks 2+2+1 / 2+2
    ("conv1d-019", dict(max_block_rows=2), ["auto", "eager"]),  # TILE: 1-row ranges are short 2-row blocks
    ("conv1d-006", {}, ["eager"]),  # L1 input, one row per core, rows migrate (L1 halo reads)
    ("conv1d-010", {}, ["eager"]),  # L1 output, rows migrate
    ("conv1d-004", dict(silu_body=QUARTIC, split_rows=16), ["auto"]),  # the recommended multi-row combination
    ("conv1d-020", dict(silu_body=QUARTIC, split_rows=16), ["auto"]),
    ("conv1d-020", dict(silu_body=QUARTIC), ["auto"]),
]
BITIDENT_PARAMS = [(c, l, m) for c, l, ms in BITIDENT_CELLS for m in ms]


@pytest.mark.parametrize(
    "case_id,limits,mode",
    BITIDENT_PARAMS,
    ids=[f"{c}-{'_'.join(f'{k}{v}' for k, v in l.items()) or 'plan'}-{m}" for c, l, m in BITIDENT_PARAMS],
)
def test_bitident(device, monkeypatch, case_id, limits, mode):
    import torch  # local: no global torch import under ttnn/ (pre-commit)

    case = CASES[case_id]
    reference, host, device_inputs, plan0 = _call(device, monkeypatch, case, limits, None)
    if host is not None:
        _check_case(case, reference, host, device_inputs)
    ref = [ttnn.to_torch(t) for t in reference]
    for _ in range(2):
        outputs, host, device_inputs, plan = _call(device, monkeypatch, case, limits, mode)
        assert plan == plan0
        if host is not None:
            _check_case(case, outputs, host, device_inputs)
        for a, b in zip(ref, outputs):
            assert torch.equal(a, ttnn.to_torch(b)), f"{case_id} {limits} {mode}: stealing changed the result"
    print(f"PLAN {case_id} {limits} {mode} -> {steal_module.resolve_steal_mode(plan0)}: {plan0}")


# variant -> (plan_limits, mode). base* = the unmodified op.
PERF_VARIANTS = {
    "base": ({}, None),
    "base_q": (dict(silu_body=QUARTIC), None),
    "steal": ({}, "auto"),
    "steal_q": (dict(silu_body=QUARTIC), "auto"),
    "own": ({}, "own"),
    "own_q": (dict(silu_body=QUARTIC), "own"),
    "on": ({}, "on"),
    "static": ({}, "static"),
}
# split-ratio companions (plan_limits(split_rows=...): the writer RISC's NoC1 share of each tile-row's reads)
for _split in (0, 8, 16):
    PERF_VARIANTS[f"base_q_s{_split}"] = (dict(silu_body=QUARTIC, split_rows=_split), None)
    PERF_VARIANTS[f"steal_q_s{_split}"] = (dict(silu_body=QUARTIC, split_rows=_split), "auto")
    PERF_VARIANTS[f"base_s{_split}"] = (dict(split_rows=_split), None)
    PERF_VARIANTS[f"steal_s{_split}"] = (dict(split_rows=_split), "auto")

PERF_CASES = [
    "conv1d-004",
    "conv1d-001",
    "conv1d-002",
    "conv1d-003",
    "conv1d-005",
    "conv1d-006",
    "conv1d-010",
    "conv1d-011",
    "conv1d-012",
    "conv1d-013",
    "conv1d-014",
    "conv1d-016",
    "conv1d-019",
    "conv1d-020",
]


@pytest.mark.skipif(os.environ.get("GQKV_STEAL_PERF") != "1", reason="perf bench: set GQKV_STEAL_PERF=1")
@pytest.mark.parametrize("case_id", PERF_CASES)
@pytest.mark.parametrize("variant", list(PERF_VARIANTS))
def test_perf(device, monkeypatch, case_id, variant):
    limits, mode = PERF_VARIANTS[variant]
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # drop anything earlier (operand uploads)
    outputs, host, device_inputs, plan = _call(device, monkeypatch, CASES[case_id], limits, mode)
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
    print(f"\nSTEAL_NS {variant}-{case_id} {' '.join(str(d) for d in durations)}")
    print(f"PLAN {case_id} {variant}: {plan}")
    if host is not None:
        _check_case(CASES[case_id], outputs, host, device_inputs)


@pytest.mark.parametrize("variant", ["base", "base_q", "steal_q_s16"])
def test_pcc(device, monkeypatch, variant):
    """Precision of a variant on the focus shape (conv1d-004): PCC / max abs error per output vs float64 torch."""
    import torch

    from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
        pytorch_reference,
    )

    limits, mode = PERF_VARIANTS[variant]
    case = CASES["conv1d-004"]
    outputs, host, _, _ = _call(device, monkeypatch, case, limits, mode)
    _, conv_input, history, taps = host
    refs, _ = pytorch_reference(conv_input, history, taps, case["inputs"][1])
    for name, out, ref in zip("qkv", outputs, refs):
        a = ttnn.to_torch(out).double().flatten()
        b = ref.double().flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(f"PCC {variant} {name} {pcc:.7f} maxabs {float((a - b).abs().max()):.4e}")


@pytest.mark.parametrize("case_id,limits", [("conv1d-004", dict(silu_body=QUARTIC, split_rows=16)), ("conv1d-019", {})])
def test_cache_hit(device, monkeypatch, case_id, limits):
    """Program-cache hits re-launch the same program: the claim semaphores must start from 0 on every launch."""
    import torch

    case = CASES[case_id]
    reference, _, _, _ = _call(device, monkeypatch, case, limits, None)
    ref = [ttnn.to_torch(t) for t in reference]
    op_module._PROGRAM_CACHE.clear()
    with monkeypatch.context() as mp:
        mp.setattr(op_module, "create_program_descriptor", steal_module.create_program_descriptor)
        with op_module.plan_limits(**limits):
            for i in range(4):  # launch 0 builds the program, 1..3 are cache hits
                if case["inputs"][3] is not None:
                    outputs = _run_and_check(device, case)
                else:
                    outputs, _, _ = _run(device, case)
                for a, b in zip(ref, outputs):
                    assert torch.equal(a, ttnn.to_torch(b)), f"{case_id} launch {i}: result changed"
    assert len(op_module._PROGRAM_CACHE) == 1
    op_module._PROGRAM_CACHE.clear()
