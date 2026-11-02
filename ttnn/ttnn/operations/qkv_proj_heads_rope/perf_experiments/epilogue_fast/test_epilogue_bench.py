# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated epilogue bench. Env: EPI_VARIANTS (ordered, comma list), EPI_ITERS, EPI_BLOCK_M, EPI_HEADS.
Prints EPI <variant> ns=<device kernel ns> pcc=<..> rms=<..> per variant; correctness is the only assert."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
from pathlib import Path

import ttnn

_spec = importlib.util.spec_from_file_location("epi_bench", Path(__file__).parent / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

VARIANTS = os.environ.get("EPI_VARIANTS", "baseline,win4,win2").split(",")
ITERS = [int(v) for v in os.environ.get("EPI_ITERS", "1,33").split(",")]
BLOCK_M = int(os.environ.get("EPI_BLOCK_M", "11"))
HEADS = int(os.environ.get("EPI_HEADS", "4"))
# Diagnostics only (never candidates): EPI_DIAG_LOFI=1 -> LoFi math, EPI_DIAG_BF16P=1 -> bf16 partials.
DIAG_LOFI = os.environ.get("EPI_DIAG_LOFI") == "1"
# Domain sweep (user-chosen fidelity; the same config runs for baseline and candidate): EPI_FIDELITY=HiFi4.
DIAG_KERNEL = int(os.environ.get("EPI_DIAG_KERNEL", "0"))
FIDELITY = getattr(ttnn.MathFidelity, os.environ.get("EPI_FIDELITY", "HiFi3"))
DIAG_BF16P = os.environ.get("EPI_DIAG_BF16P") == "1"
_KEY = "DEVICE KERNEL DURATION [ns]"


def _kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total = None
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_KEY)
            if entry is not None:
                total = (total or 0.0) + float(entry.duration)
    return total


def _pcc(a, b):
    import torch

    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def _rms(a, b):
    d = a.double() - b.double()
    return (d.pow(2).mean().sqrt() / b.double().pow(2).mean().sqrt()).item()


def test_epilogue_bench(device):
    import torch

    partials, table, cos, sin = bench.make_host(BLOCK_M, HEADS)
    outputs = {}
    for variant in VARIANTS:
        rotary = not variant.startswith("v_")
        ref = bench.reference(partials, cos, sin, HEADS, rotary)
        line = [f"EPI {variant}"]
        for iters in ITERS:
            p, t, out = bench.device_tensors(
                device, partials, table, BLOCK_M, HEADS, ttnn.bfloat16 if DIAG_BF16P else ttnn.float32
            )
            cfg = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi if DIAG_LOFI else FIDELITY,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            )
            desc = bench.descriptor(
                p,
                t,
                out,
                variant=variant,
                block_m=BLOCK_M,
                heads=HEADS,
                iters=iters,
                compute_config=cfg,
                diag=DIAG_KERNEL,
            )
            ttnn.generic_op([p, t, out], desc)
            ns = _kernel_ns(device)
            got = ttnn.to_torch(out).float()
            pcc, rms = _pcc(got, ref), _rms(got, ref)
            line.append(f"iters={iters} ns={ns:.0f} pcc={pcc:.6f} rms={rms:.3e}")
            assert torch.isfinite(got).all()
            assert (
                (DIAG_LOFI or DIAG_BF16P or DIAG_KERNEL) or pcc >= 0.9999 and rms <= 4e-3
            ), f"{variant}: pcc {pcc} rms {rms}"
            outputs.setdefault(variant, got)
        if len(ITERS) == 2:
            (i0, n0), (i1, n1) = [(ITERS[k], float(line[k + 1].split("ns=")[1].split()[0])) for k in range(2)]
            line.append(f"per_iter_ns={(n1 - n0) / (i1 - i0):.0f}")
        print(" | ".join(line))
    base = outputs.get("baseline")
    for variant, got in outputs.items():
        if base is not None and not variant.startswith("v_"):
            print(
                f"EPI {variant} bitwise_equal_baseline={torch.equal(got, base)} "
                f"max_abs_diff={(got - base).abs().max().item():.3e}"
            )
