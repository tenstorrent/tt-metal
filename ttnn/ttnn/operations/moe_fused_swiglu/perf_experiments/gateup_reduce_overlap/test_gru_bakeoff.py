# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gateup_reduce_overlap — isolated bake-off: does pipelining the gate/up matmul against the
cross-column reduce (op_design.md §4.3, never built — changelog.md Refinement 2 lever 2 parked it
because splitting the matmul ALONE shrinks DEST without buying the overlap) beat the op's honest
current approach (whole-block matmul, then whole-block reduce)?

One fresh-cache `ttnn.generic_op` launch per variant/shape point — device kernel time has no
warm-up transient (see /perf-measure), so this reads `DEVICE KERNEL DURATION [ns]` via the in-process
profiler exactly once per case (matching examples/matmul_output_subblock's measurement pattern) and
prints ns + PCC for every case. Correctness is a hard gate (assert_with_pcc); perf is reported, never
asserted, per the perf-part-optimizer protocol.

Run:
    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/gateup_reduce_overlap/test_gru_bakeoff.py
"""

import os

# In-process device profiler (before ttnn import) — see examples/matmul_output_subblock's test.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest

# NOTE: `torch` is imported LAZILY here. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/` so that importing ttnn
# never drags torch in. These perf-experiment benches live under the op directory, so they
# obey the same rule: every use sites gets `import torch` inside the function.
import ttnn

from ttnn.operations.moe_fused_swiglu.perf_experiments.gateup_reduce_overlap.gru_program_descriptor import (
    KGROUPS,
    kr_sizes_starts,
    make_x_tensor,
    make_weight_tensor,
    create_program_descriptor,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
PCC_THRESHOLD = 0.975  # soft gate given by the coordinator; real op's measured bfp4 floor ~0.9797


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    import torch

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b).item() / denom)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _quantized_reference(x_bf8_torch, wg_bf4_torch, wu_bf4_torch):
    """SiLU(x @ w_gate) * (x @ w_up) in fp32, over tensors already round-tripped through the SAME
    bfp8_b (x) / bfp4_b (weights) quantization the kernel reads — this measures the KERNEL's
    schedule, not the format floor (matches the real op's own precision_baseline methodology)."""
    import torch

    x = x_bf8_torch.to(torch.float32)
    wg = wg_bf4_torch.to(torch.float32)
    wu = wu_bf4_torch.to(torch.float32)
    gate = x @ wg
    up = x @ wu
    return torch.nn.functional.silu(gate) * up


def _build_inputs(device, emb_t, hn_pad, m_eff, seed=0):
    import torch

    torch.manual_seed(seed)
    k = emb_t * TILE
    x_torch = torch.randn(m_eff * TILE, k, dtype=torch.float32) * 0.5
    wg_torch = torch.randn(k, hn_pad * TILE, dtype=torch.float32) * 0.5
    wu_torch = torch.randn(k, hn_pad * TILE, dtype=torch.float32) * 0.5

    tt_x = make_x_tensor(x_torch, device)
    tt_wg = make_weight_tensor(wg_torch, device)
    tt_wu = make_weight_tensor(wu_torch, device)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m_eff * TILE, hn_pad * TILE]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # The reference must see the SAME quantization the kernel reads (bfp8_b x, bfp4_b weights) —
    # round-trip through the device, exactly like moe_fused_swiglu's own format-floor probes.
    x_q = ttnn.to_torch(tt_x)
    wg_q = ttnn.to_torch(tt_wg)
    wu_q = ttnn.to_torch(tt_wu)
    return tt_x, tt_wg, tt_wu, tt_out, x_q, wg_q, wu_q


def _run_case(device, *, emb_t=224, hn_pad=6, m_eff=8, split_axis="hn", s_stages=1, pipelined=False, seed=0):
    import torch

    tt_x, tt_wg, tt_wu, tt_out, x_q, wg_q, wu_q = _build_inputs(device, emb_t, hn_pad, m_eff, seed=seed)
    desc = create_program_descriptor(
        tt_x,
        tt_wg,
        tt_wu,
        tt_out,
        device=device,
        emb_t=emb_t,
        hn_pad=hn_pad,
        m_eff=m_eff,
        s_stages=s_stages,
        split_axis=split_axis,
        pipelined=pipelined,
    )
    ttnn.generic_op([tt_x, tt_wg, tt_wu, tt_out], desc)
    ns = _read_kernel_ns(device)
    out_torch = ttnn.to_torch(tt_out).to(torch.float32)
    ref = _quantized_reference(x_q, wg_q, wu_q)
    pcc = _pcc(ref, out_torch)
    return ns, pcc


# ---------------------------------------------------------------------------
# Correctness: every variant must compute the identical SiLU(gate)*up sum, regardless of how the
# gate/up matmul + reduce is scheduled or split.
# ---------------------------------------------------------------------------
CORRECTNESS_CASES = [
    ("baseline", "hn", 1, False, 6, 8),
    ("hn_s2_serial", "hn", 2, False, 6, 8),
    ("hn_s2_pipelined", "hn", 2, True, 6, 8),
    ("hn_s3_pipelined", "hn", 3, True, 6, 8),
    ("hn_s6_pipelined", "hn", 6, True, 6, 8),
    ("m_s2_pipelined", "m", 2, True, 6, 8),
    ("m_s4_pipelined", "m", 4, True, 6, 8),
    # m_s8_pipelined dropped: fails to build ("Program size (86384) too large for kernel config
    # buffer (70656)") — the 8-way compile-time-unrolled StageRunner blows the kernel-config binary
    # size limit. A real, reportable constraint on how deep this idea's pipeline can go, not a
    # correctness bug in the idea itself.
    ("hn_pad4_baseline", "hn", 1, False, 4, 8),
    ("hn_pad4_s2_pipelined", "hn", 2, True, 4, 8),
    ("m_eff4_baseline", "hn", 1, False, 6, 4),
    ("m_eff4_hn_s2_pipelined", "hn", 2, True, 6, 4),
    ("m_eff4_m_s2_pipelined", "m", 2, True, 6, 4),
    ("m_eff1_baseline", "hn", 1, False, 6, 1),
    ("m_eff1_hn_s2_pipelined", "hn", 2, True, 6, 1),
]


@pytest.mark.parametrize("name, axis, s, pipe, hn_pad, m_eff", CORRECTNESS_CASES)
def test_correctness(device, name, axis, s, pipe, hn_pad, m_eff):
    ns, pcc = _run_case(device, hn_pad=hn_pad, m_eff=m_eff, split_axis=axis, s_stages=s, pipelined=pipe)
    print(f"[gru][correctness] {name}: ns={ns:.0f} pcc={pcc:.5f}")
    assert pcc >= PCC_THRESHOLD, f"{name}: pcc {pcc:.5f} below threshold {PCC_THRESHOLD}"


# ---------------------------------------------------------------------------
# Device-ns bake-off. One fresh run per case (no warm-up loop: device kernel time has no warm-up
# transient). Perf is reported, never asserted.
# ---------------------------------------------------------------------------
FOCUS_CASES = [
    ("baseline", "hn", 1, False, 6, 8),
    ("hn_s2_serial", "hn", 2, False, 6, 8),
    ("hn_s2_pipelined", "hn", 2, True, 6, 8),
    ("hn_s3_serial", "hn", 3, False, 6, 8),
    ("hn_s3_pipelined", "hn", 3, True, 6, 8),
    ("hn_s6_pipelined", "hn", 6, True, 6, 8),
    ("m_s2_serial", "m", 2, False, 6, 8),
    ("m_s2_pipelined", "m", 2, True, 6, 8),
    ("m_s4_pipelined", "m", 4, True, 6, 8),
    ("m_s4_serial", "m", 4, False, 6, 8),
]


@pytest.mark.parametrize("name, axis, s, pipe, hn_pad, m_eff", FOCUS_CASES)
def test_device_ns_focus(device, name, axis, s, pipe, hn_pad, m_eff):
    ns, pcc = _run_case(device, hn_pad=hn_pad, m_eff=m_eff, split_axis=axis, s_stages=s, pipelined=pipe)
    print(f"[gru][focus emb7168/hn{hn_pad}/m_eff{m_eff}] {name}: ns={ns:.0f} pcc={pcc:.5f}")


PREDICATE_CASES = [
    # m_eff sweep (count 128 / count 32 regimes)
    ("m_eff4_baseline", "hn", 1, False, 6, 4),
    ("m_eff4_hn_s2_pipelined", "hn", 2, True, 6, 4),
    ("m_eff4_hn_s3_pipelined", "hn", 3, True, 6, 4),
    ("m_eff4_m_s2_pipelined", "m", 2, True, 6, 4),
    ("m_eff4_m_s4_pipelined", "m", 4, True, 6, 4),
    ("m_eff1_baseline", "hn", 1, False, 6, 1),
    ("m_eff1_hn_s2_pipelined", "hn", 2, True, 6, 1),
    ("m_eff1_hn_s3_pipelined", "hn", 3, True, 6, 1),
    # ragged hidden column (hn_pad=4, real grid's column x=10)
    ("hn_pad4_baseline", "hn", 1, False, 4, 8),
    ("hn_pad4_s2_pipelined", "hn", 2, True, 4, 8),
    ("hn_pad4_s4_pipelined", "hn", 4, True, 4, 8),
    ("hn_pad4_m_s2_pipelined", "m", 2, True, 4, 8),
]


@pytest.mark.parametrize("name, axis, s, pipe, hn_pad, m_eff", PREDICATE_CASES)
def test_device_ns_predicate_sweep(device, name, axis, s, pipe, hn_pad, m_eff):
    ns, pcc = _run_case(device, hn_pad=hn_pad, m_eff=m_eff, split_axis=axis, s_stages=s, pipelined=pipe)
    print(f"[gru][sweep hn{hn_pad}/m_eff{m_eff}] {name}: ns={ns:.0f} pcc={pcc:.5f}")
