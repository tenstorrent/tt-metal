# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: moe_fused_swiglu's reduce-ROOT epilogue fusion.

h = SiLU(gate_acc + last_gate_child) * (up_acc + last_up_child), single core, compute-only
(sharded L1, no NoC transport). Correctness is the only pass/fail; perf is measured (DEVICE
KERNEL DURATION [ns] via ReadDeviceProfiler) and reported, never asserted. See the module
docstring in program_descriptor_with_inline_kernels.py for the full variant menu and the DEST
accounting behind it.

Reference: the op's own measured bfp4 format floor is ~0.9797, soft PCC gate 0.975
(moe_fused_swiglu/feature_spec.py / changelog.md). Since this bench carries only ONE bfp8
requantization hop (not three bfp4 matmuls), PCC should sit far above that floor for every
variant that respects the precision contract.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

# NOTE: `torch` is imported LAZILY here. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/` so that importing ttnn
# never drags torch in. These perf-experiment benches live under the op directory, so they
# obey the same rule: every use sites gets `import torch` inside the function.
import ttnn
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.root_epilogue_fusion.program_descriptor_with_inline_kernels import (
    BASELINE,
    VARIANTS,
    create_sharded_memory_config,
    run_op,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
PCC_THRESHOLD = 0.975  # feature_spec.py pcc_threshold (soft gate)

# =============================================================================
# Reference construction — bfp8-dequantized inputs (isolates KERNEL correctness from the
# bfloat8_b format's own quantization, matching the op's own precision-baseline methodology).
# =============================================================================


def _make_case(device, m_eff, hn_pad, seed=7):
    import torch

    torch.manual_seed(seed)
    m, n = m_eff * TILE, hn_pad * TILE
    mem_cfg = create_sharded_memory_config((m, n))

    def _dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg)

    gate_acc = torch.randn(m, n) * 0.5
    up_acc = torch.randn(m, n) * 0.5
    reduce_gate_in = torch.randn(m, n) * 0.5
    reduce_up_in = torch.randn(m, n) * 0.5

    gate_acc_dev = _dev(gate_acc)
    up_acc_dev = _dev(up_acc)
    reduce_gate_in_dev = _dev(reduce_gate_in)
    reduce_up_in_dev = _dev(reduce_up_in)

    # Dequantize by reading back what actually landed in bfloat8_b, so the reference isolates
    # kernel-attributable error from the format's own rounding (op_design.md's own approach).
    g0 = ttnn.to_torch(gate_acc_dev).to(torch.float32)
    u0 = ttnn.to_torch(up_acc_dev).to(torch.float32)
    g1 = ttnn.to_torch(reduce_gate_in_dev).to(torch.float32)
    u1 = ttnn.to_torch(reduce_up_in_dev).to(torch.float32)
    expected = torch.nn.functional.silu(g0 + g1) * (u0 + u1)

    inputs = [gate_acc_dev, up_acc_dev, reduce_gate_in_dev, reduce_up_in_dev]
    return inputs, expected


def _pcc(actual, expected):
    import torch

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    if torch.all(a == a[0]) or torch.all(e == e[0]):
        return 1.0 if torch.allclose(a, e) else 0.0
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def _check(output, expected, label, min_pcc=PCC_THRESHOLD):
    import torch

    actual = ttnn.to_torch(output).to(torch.float32)
    pcc = _pcc(actual, expected)
    assert pcc >= min_pcc, f"{label}: PCC {pcc:.5f} < {min_pcc}"
    return pcc


# =============================================================================
# In-process device-kernel timing (validated pattern from examples/compute_block_size — do not
# reinvent).
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warm-up window
    samples = {name: [] for name in runners}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {name}"
            if trial:  # discard first timed pass
                samples[name].append(duration / kernel_iters)
    return samples


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


# =============================================================================
# Correctness — every variant, one shot each, against the same fixture.
# =============================================================================
def test_root_epilogue_fusion_correctness(device):
    for m_eff, hn_pad in ((8, 6), (4, 6), (1, 6), (8, 4)):
        inputs, expected = _make_case(device, m_eff, hn_pad)
        for variant in VARIANTS:
            out = run_op(inputs, m_eff=m_eff, hn_pad=hn_pad, variant=variant, kernel_iters=1)
            pcc = _check(out, expected, f"{variant} m_eff={m_eff} hn_pad={hn_pad}")
            logger.info(f"{variant:16s} m_eff={m_eff} hn_pad={hn_pad}  PCC={pcc:.6f}")


# =============================================================================
# Perf — focus shape (m_eff=8, hn_pad=6, the count-256 regime) + predicate sweep.
# =============================================================================
def _int(name, default):
    return int(os.environ.get(name, default))


def test_root_epilogue_fusion_device_perf(device):
    if True:
        trials = _int("REF_TRIALS", "3")
        kernel_iters = _int("REF_KERNEL_ITERS", "30")

        regimes = [
            ("focus (count 256)", 8, 6),
            ("count 128", 4, 6),
            ("count 32", 1, 6),
            ("ragged column (hn_pad=4)", 8, 4),
        ]

        report_lines = [
            "# root_epilogue_fusion — isolated bake-off (single core, compute-only)",
            "",
            f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  "
            f"placement=single-core sharded-L1  N={trials} (median)  kernel-iters={kernel_iters}",
            "",
            "Metric: DEVICE KERNEL DURATION [ns] per root-epilogue evaluation "
            "(h = SiLU(gate_acc+last_gate_child) * (up_acc+last_up_child)).",
            "",
        ]

        all_samples = {}
        all_pccs = {}
        for label, m_eff, hn_pad in regimes:
            inputs, expected = _make_case(device, m_eff, hn_pad)

            pccs = {}
            for variant in VARIANTS:
                out = run_op(inputs, m_eff=m_eff, hn_pad=hn_pad, variant=variant, kernel_iters=1)
                pccs[variant] = _check(out, expected, f"{variant} {label}")

            runners = {
                variant: (
                    lambda m_eff=m_eff, hn_pad=hn_pad, variant=variant: run_op(
                        inputs, m_eff=m_eff, hn_pad=hn_pad, variant=variant, kernel_iters=kernel_iters
                    )
                )
                for variant in VARIANTS
            }
            samples = _measure(device, runners, trials, kernel_iters)
            all_samples[label] = samples
            all_pccs[label] = pccs

            base_med = statistics.median(samples[BASELINE])
            report_lines.append(f"## {label}  (m_eff={m_eff}, hn_pad={hn_pad}, block_tiles={m_eff * hn_pad})")
            report_lines.append("")
            report_lines.append("| Variant | Median ns | Speedup vs baseline | PCC |")
            report_lines.append("|---|---:|---:|---:|")
            for variant in VARIANTS:
                med = statistics.median(samples[variant])
                speedup = base_med / med
                report_lines.append(f"| {variant} | {med:.1f} | {speedup:.2f}x | {pccs[variant]:.6f} |")
            report_lines.append("")
            logger.info(f"{label}: baseline={base_med:.1f}ns")

        report = "\n".join(report_lines) + "\n"
        logger.info("\n" + report)
        if report_path := os.environ.get("REF_REPORT"):
            Path(report_path).write_text(report)
