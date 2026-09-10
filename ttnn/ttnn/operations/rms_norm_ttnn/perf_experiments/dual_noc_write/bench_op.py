# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ---------------------------------------------------------------------------
# dual_noc_write -- REAL-OP A/B for the single-RISC dual-NoC output write.
# ---------------------------------------------------------------------------
# The isolated DM instrument (dm_bench.py) showed the shipped reader-NoC0 /
# writer-NoC1 pairing is already optimal, but that NoC1 is the longer pole and
# NoC0 has spare time; moving ~10 % of the output write onto NoC0 measured
# 1.02-1.03x on the pure transfer.  This file prices that in the REAL op.
#
# Everything is held constant except the two variables:
#   * which kernel directory the three sources come from (PD.KERNEL_DIR)
#   * whether the two data-movement kernels are built DM_DEDICATED_NOC (shipped)
#     or DM_DYNAMIC_NOC (required for one RISC-V to touch both NoCs)
# The precision contract (HiFi2 / fp32_dest_acc_en=False / math_approx=False) and
# every dtype are IDENTICAL in every variant -- they are never a lever.
#
#   base  : shipped kernels, shipped configs.
#   kbase : k_base (a byte-identical copy of the shipped kernels), shipped
#           configs.  A control: it must agree with `base`.
#   dyn   : k_base + DM_DYNAMIC_NOC.  Prices the dynamic-NoC mode alone.
#   alt   : k_alt  + DM_DYNAMIC_NOC.  THE CANDIDATE -- the writer issues the last
#           WT_CHUNK/10 tiles of every row-block on the OTHER NoC.

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
from pathlib import Path

import ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config

HERE = Path(__file__).resolve().parent
_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_TRIALS = int(os.environ.get("RMS_TRIALS", "3"))

_SHIPPED_KERNEL_DIR = PD.KERNEL_DIR
_SHIPPED_READER_CFG = PD._reader_dm_config
_SHIPPED_WRITER_CFG = PD._writer_dm_config

_DYN = ttnn.NOC_MODE.DM_DYNAMIC_NOC


def _dyn_reader_cfg(plan):
    cfg = _SHIPPED_READER_CFG(plan)
    noc = ttnn.NOC.NOC_0 if isinstance(cfg, ttnn.ReaderConfigDescriptor) else cfg.noc
    return ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=noc, noc_mode=_DYN)


def _dyn_writer_cfg(plan):
    cfg = _SHIPPED_WRITER_CFG(plan)
    noc = ttnn.NOC.NOC_1 if isinstance(cfg, ttnn.WriterConfigDescriptor) else cfg.noc
    return ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=noc, noc_mode=_DYN)


# label -> (kernel_dir or None, dynamic_noc)
VARIANTS = {
    "base": (None, False),
    "kbase": (HERE / "k_base", False),
    "dyn": (HERE / "k_base", True),
    "alt": (HERE / "k_alt", True),
}

# name: (shape, shard|None, memory_layout)
CASES = {
    "FOCUS_8192x2304": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED),
    "I_8192x1024": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED),
    "I_8192x7168": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED),
    "S_32x1024": ((1, 1, 32, 1024), None, _ML.INTERLEAVED),
    "W_32x7168_shard": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED),
}


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


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def build(device, name):
    import torch

    shape, shard, ml = CASES[name]
    torch.manual_seed(0)
    W = shape[-1]
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    if shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    torch.manual_seed(1)
    tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
    g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _cfg(), "memory_config": x.memory_config(), "weight": g}
    expected = torch_rms_norm_ttnn(tx.float(), epsilon=1e-12, weight=tg.float())
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, [x, g]


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def measure(device, name):
    run, expected, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out)
    p = pcc(got, expected)
    del out, got
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_TRIALS):
        run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        if v is not None:
            samples.append(v)
    ns = statistics.median(samples) if samples else float("nan")
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, p


def sweep(labels, names, device=None):
    RES = {}
    own = device is None
    if own:
        device = ttnn.open_device(device_id=0)
    try:
        for label in labels:
            kdir, dyn = VARIANTS[label]
            PD.KERNEL_DIR = _SHIPPED_KERNEL_DIR if kdir is None else kdir
            PD._reader_dm_config = _dyn_reader_cfg if dyn else _SHIPPED_READER_CFG
            PD._writer_dm_config = _dyn_writer_cfg if dyn else _SHIPPED_WRITER_CFG
            try:
                for name in names:
                    ns, p = measure(device, name)
                    RES[(name, label)] = (ns, p)
                    print(f"RESULT.raw {name:18s} {label:8s} ns={ns:11.0f} pcc={p:.6f}")
            finally:
                PD.KERNEL_DIR = _SHIPPED_KERNEL_DIR
                PD._reader_dm_config = _SHIPPED_READER_CFG
                PD._writer_dm_config = _SHIPPED_WRITER_CFG
        print("RESULT " + f"{'case':18s}" + "".join(f"{l:>12s}" for l in labels))
        for name in names:
            print("RESULT " + f"{name:18s}" + "".join(f"{RES[(name, l)][0]:12.0f}" for l in labels))
        print("RESULT --- speedup vs " + labels[0] + " (>1 = faster) ---")
        for name in names:
            b = RES[(name, labels[0])][0]
            row = "".join(f"{b / RES[(name, l)][0]:12.3f}" for l in labels)
            worst = min(RES[(name, l)][1] for l in labels)
            print("RESULT " + f"{name:18s}" + row + f"   worst_pcc={worst:.6f}")
    finally:
        if own:
            ttnn.close_device(device)
    return RES
