# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf experiment `dynamic_split_reads` — bit-identity gate + device-time bench (measurement only).

baseline  = the unmodified op (static split: the writer RISC reads body rows [8, 32) of every tile-row)
candidate = the same op with create_program_descriptor swapped for dyn_program_descriptor's
            (reader / writer claim each tile-row's 32 sticks dynamically; compute kernel unchanged)

Correctness:  scripts/run_safe_pytest.sh --no-jit-server --run-all <this file> -k bitident
Perf (one op call per test; DEVICE KERNEL DURATION read in-process with ttnn.ReadDeviceProfiler, printed as
"DYN_NS <variant>-<case> <ns>"; the device log with the stage zones lands in $TT_METAL_PROFILER_DIR/.logs):
    run_perf.sh <variant>-<case> ...      (sets TT_METAL_DEVICE_PROFILER=1 etc.; see there)
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
dyn_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.dynamic_split_reads.dyn_program_descriptor"
)

CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES}
CASES["rmT256"] = _case("rmT256", ((1, 256, 2560), (512, 512, 1536), 256, None, 0), ttnn.ROW_MAJOR_LAYOUT).values[0]
QUARTIC = desc_module.SILU_BF16_QUARTIC


def _call(device, monkeypatch, case, limits, knobs):
    """One op call. knobs None -> the unmodified op; a dict -> the dynamic split with those knobs."""
    op_module._PROGRAM_CACHE.clear()
    with monkeypatch.context() as mp:
        if knobs is not None:
            for name, value in knobs.items():
                mp.setattr(dyn_module, name, value)
            mp.setattr(op_module, "create_program_descriptor", dyn_module.create_program_descriptor)
        with op_module.plan_limits(**limits):
            if case["inputs"][3] is not None:  # channel-paged history: own operand builder (+ checks)
                outputs, host, device_inputs = _run_and_check(device, case), None, None
            else:
                outputs, host, device_inputs = _run(device, case)
            plan = op_module.get_last_plan()
    op_module._PROGRAM_CACHE.clear()
    return outputs, host, device_inputs, plan


DEFAULT = {}
STRESS = dict(READ_WINDOW=1, WRITER_WINDOW=1, CLAIM_CHUNK=1, MIN_ENTER=1, POLL_IN_FLUSH=True, TAIL_WINDOW=1)
WIDE = dict(READ_WINDOW=32, WRITER_WINDOW=32, CLAIM_CHUNK=4, MIN_ENTER=1, TAIL_WINDOW=32, GATE_WRITES=True)
# speculative push (re-read the writer's share at once) with a tail left for the writer: maximizes duplicates,
# late writer reads and slot-reuse guard waits
SPEC = dict(READ_WINDOW=32, WRITER_WINDOW=8, CLAIM_CHUNK=1, HEAD_STICKS=8, TAIL_WINDOW=1, DUP_GRACE=0)

# (case, plan_limits) cells: the flagged cells + the split-reader edge cases of test_split_reader.py
BITIDENT_CELLS = [
    ("conv1d-004", {}),
    ("conv1d-004", dict(silu_body=QUARTIC)),
    ("conv1d-001", {}),
    ("conv1d-002", {}),
    ("conv1d-003", {}),
    ("conv1d-003", dict(max_cores_per_group=2)),  # DRAM halo at t0 > 0
    ("conv1d-006", {}),
    ("conv1d-006", dict(max_cores_per_group=1)),  # L1 input, multi-block
    ("conv1d-010", {}),
    ("conv1d-010", dict(max_block_rows=1, max_cores_per_group=1)),  # L1 output, multi-block
    ("conv1d-011", {}),
    ("conv1d-012", {}),
    ("conv1d-013", {}),
    ("conv1d-014", {}),
    ("conv1d-016", {}),
    ("conv1d-020", {}),
    ("rmT256", dict(max_block_rows=3, max_cores_per_group=1)),  # 3-row blocks, ragged tail
    ("rmT256", dict(max_block_rows=4, max_cores_per_group=2)),  # 4-row blocks, DRAM halo
    ("conv1d-005", {}),  # TILE regime: split off -> the dyn kernels' non-split path
]


@pytest.mark.parametrize("knobs_name", ["default", "stress", "wide", "spec"])
@pytest.mark.parametrize(
    "case_id,limits",
    BITIDENT_CELLS,
    ids=[f"{c}-{'_'.join(f'{k}{v}' for k, v in l.items()) or 'plan'}" for c, l in BITIDENT_CELLS],
)
def test_bitident(device, monkeypatch, case_id, limits, knobs_name):
    import torch  # local: no global torch import under ttnn/ (pre-commit)

    knobs = {"default": DEFAULT, "stress": STRESS, "wide": WIDE, "spec": SPEC}[knobs_name]
    case = CASES[case_id]
    reference, host, device_inputs, plan0 = _call(device, monkeypatch, case, limits, None)
    if host is not None:
        _check_case(case, reference, host, device_inputs)
    ref = [ttnn.to_torch(t) for t in reference]
    repeats = 3 if knobs_name == "default" else 2
    for _ in range(repeats):
        outputs, host, device_inputs, plan = _call(device, monkeypatch, case, limits, knobs)
        assert plan == plan0
        if host is not None:
            _check_case(case, outputs, host, device_inputs)
        for a, b in zip(ref, outputs):
            assert torch.equal(
                a, ttnn.to_torch(b)
            ), f"{case_id} {limits} {knobs_name}: dynamic split changed the result"
    print(f"PLAN {case_id} {limits}: {plan0}")


PERF_VARIANTS = {
    "base": ({}, None),
    "dyn": ({}, DEFAULT),
    "base_q": (dict(silu_body=QUARTIC), None),
    "dyn_q": (dict(silu_body=QUARTIC), DEFAULT),
}
# knob sweep entries: dyn_r<READ>_w<WRITER>_c<CHUNK>[_f: poll in flush][_g: gate on unacked writes]
# (+ _q for the quartic body)
for r_win in (4, 8, 16, 32):
    for w_win in (1, 2, 4, 8, 16, 32):
        for chunk in (1, 2, 4):
            for flush in (False, True):
                for gate in (False, True):
                    for q in (False, True):
                        if chunk > min(r_win, w_win):
                            continue
                        name = (
                            f"dyn_r{r_win}_w{w_win}_c{chunk}"
                            + ("_f" if flush else "")
                            + ("_g" if gate else "")
                            + ("_q" if q else "")
                        )
                        PERF_VARIANTS[name] = (
                            dict(silu_body=QUARTIC) if q else {},
                            dict(
                                READ_WINDOW=r_win,
                                WRITER_WINDOW=w_win,
                                CLAIM_CHUNK=chunk,
                                POLL_IN_FLUSH=flush,
                                GATE_WRITES=gate,
                            ),
                        )

# head/tail entries: dyn_h<HEAD>_t<TAIL_WINDOW>_r<READ>_w<WRITER>[_g][_q], chunk 2
for head in (0, 4, 8, 12, 16):
    for tail in (1, 2, 4, 8):
        for r_win in (8, 16, 32):
            for w_win in (4, 8, 16, 32):
                for gate in (False, True):
                    for q in (False, True):
                        name = f"dyn_h{head}_t{tail}_r{r_win}_w{w_win}" + ("_g" if gate else "") + ("_q" if q else "")
                        PERF_VARIANTS[name] = (
                            dict(silu_body=QUARTIC) if q else {},
                            dict(
                                READ_WINDOW=r_win,
                                WRITER_WINDOW=w_win,
                                CLAIM_CHUNK=2 if tail >= 2 else 1,
                                HEAD_STICKS=head,
                                TAIL_WINDOW=tail,
                                GATE_WRITES=gate,
                            ),
                        )

# speculative entries: <base name>_s<DUP_GRACE polls> on top of a few base configs
for base_name in list(PERF_VARIANTS):
    if not base_name.startswith(
        (
            "dyn_h8_t1_r32_w8_g",
            "dyn_h8_t2_r32_w8_g",
            "dyn_h0_t",
            "dyn_h16_t1_r32_w8_g",
            "dyn_r32_w8_c2_g",
            "dyn_r32_w4_c2",
            "dyn_h8_t1_r32_w16_g",
            "dyn_h4_t1_r32_w8_g",
        )
    ):
        continue
    for grace in (0, 64, 256, 1024):
        lim, kn = PERF_VARIANTS[base_name]
        q = base_name.endswith("_q")
        name = (base_name[:-2] if q else base_name) + f"_s{grace}" + ("_q" if q else "")
        PERF_VARIANTS[name] = (lim, dict(kn, DUP_GRACE=grace))


@pytest.mark.skipif(os.environ.get("GQKV_DYN_PERF") != "1", reason="perf bench: set GQKV_DYN_PERF=1")
@pytest.mark.parametrize(
    "case_id",
    [
        "conv1d-004",
        "conv1d-001",
        "conv1d-002",
        "conv1d-003",
        "conv1d-020",
        "conv1d-006",
        "conv1d-010",
        "conv1d-011",
        "conv1d-012",
        "conv1d-013",
        "conv1d-005",
        "conv1d-019",
    ],
)
@pytest.mark.parametrize("variant", list(PERF_VARIANTS))
def test_perf(device, monkeypatch, case_id, variant):
    limits, knobs = PERF_VARIANTS[variant]
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # drop anything earlier (operand uploads)
    outputs, host, device_inputs, plan = _call(device, monkeypatch, CASES[case_id], limits, knobs)
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
    print(f"\nDYN_NS {variant}-{case_id} {' '.join(str(d) for d in durations)}")
    print(f"PLAN {case_id} {variant}: {plan}")
    if host is not None:
        _check_case(CASES[case_id], outputs, host, device_inputs)
