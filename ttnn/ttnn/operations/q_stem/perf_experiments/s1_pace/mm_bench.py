# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""s1_pace: the stage-1 matmul per block in isolation (one core, single chip).

q block (M x cols) = x (M x K bf16) @ q_b slice (K x cols bfp8_b), HiFi2, 16-bit DEST (PERF FOCUS contract),
NB blocks back to back with x already in L1 (the reader re-pushes two filled slots). Per-block ns = (t(NB_HI) -
t(NB_LO)) / (NB_HI - NB_LO), both DEVICE KERNEL DURATION, so the fixed q_b / x load and launch cancel.

    MM_VARIANTS=helper,raw MM_M=1 MM_COLS=2 MM_K=48 scripts/run_safe_pytest.sh --no-precompile \\
        ttnn/ttnn/operations/q_stem/perf_experiments/s1_pace/mm_bench.py -s
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import json
import socket
import time
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
DUR = "DEVICE KERNEL DURATION [ns]"
VARIANTS = {"helper": 0, "helper_hoist": 1, "raw": 2, "raw_ct1": 3}
NAMES = [n for n in os.environ.get("MM_VARIANTS", ",".join(VARIANTS)).split(",") if n]
M = int(os.environ.get("MM_M", "1"))
COLS = int(os.environ.get("MM_COLS", "2"))
K = int(os.environ.get("MM_K", "48"))
NB_LO, NB_HI = 4, int(os.environ.get("MM_NB_HI", "44"))
FID = getattr(ttnn.MathFidelity, os.environ.get("MM_FIDELITY", "HiFi2"))
FP32 = os.environ.get("MM_FP32", "0") == "1"
QB_DTYPE = getattr(ttnn, os.environ.get("MM_QB_DTYPE", "bfloat8_b"))
XR = 4
CB_X, CB_QB, CB_OUT = 0, 1, 2
OUT = HERE / "results" / "mm_bench.jsonl"


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data() or {}
    vals = []
    for programs in data.values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(DUR)
            if e is not None:
                vals.append(float(e.duration))
    return vals[-1] if vals else None


def _cb(index, dtype, page, pages, crs):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _run(device, x, qb, out, variant, nb):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    xt, qt, ot = x.buffer_aligned_page_size(), qb.buffer_aligned_page_size(), out.buffer_aligned_page_size()
    sbw = COLS if M * COLS <= (4 if FP32 else 8) else 1
    cbs = [
        _cb(CB_X, x.dtype, xt, 2 * M * K, crs),
        _cb(CB_QB, qb.dtype, qt, K * COLS, crs),
        _cb(CB_OUT, out.dtype, ot, nb * M * COLS, crs),
    ]
    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_rt[0][0] = [x.buffer_address(), qb.buffer_address()]
    wr_rt[0][0] = [out.buffer_address()]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "mm_reader.cpp"),
        core_ranges=crs,
        compile_time_args=[CB_X, CB_QB, K, COLS, M, nb, XR]
        + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(qb).get_compile_time_args()),
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "mm_writer.cpp"),
        core_ranges=crs,
        compile_time_args=[CB_OUT, nb * M * COLS] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "mm_compute.cpp"),
        core_ranges=crs,
        compile_time_args=[CB_X, CB_QB, CB_OUT, K, COLS, M, nb, sbw],
        defines=[("MM_V", str(VARIANTS[variant]))],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ComputeConfigDescriptor(math_fidelity=FID, fp32_dest_acc_en=FP32, math_approx_mode=False),
    )
    desc = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs, semaphores=[])
    ttnn.generic_op([x, qb, out], desc)
    ttnn.synchronize_device(device)
    return _read_ns(device)


@pytest.mark.parametrize("name", NAMES)
def test_mm(device, name):
    import torch

    torch.manual_seed(0)
    xs = torch.randn(XR * 32, K * 32).to(torch.bfloat16)
    qbs = (torch.randn(K * 32, COLS * 32) / (K * 32) ** 0.5).to(torch.bfloat16)
    x = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    qb = ttnn.from_torch(qbs, dtype=QB_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    qb_ref = ttnn.to_torch(qb).float()  # the bfp8-rounded weights
    res = {}
    for nb in (NB_LO, NB_HI):
        out = ttnn.from_torch(
            torch.zeros(nb * M * 32, COLS * 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        _read_ns(device)
        ns = _run(device, x, qb, out, name, nb)
        got = ttnn.to_torch(out).float().view(nb, M, 32, COLS * 32)
        rows = torch.tensor([[(j % 2 * M + i) % XR for i in range(M)] for j in range(nb)])
        ref = (xs.float().view(XR, 32, K * 32)[rows] @ qb_ref).view(nb, M, 32, COLS * 32)
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        res[nb] = (ns, pcc, got)
    per_block = (res[NB_HI][0] - res[NB_LO][0]) / (NB_HI - NB_LO)
    rec = dict(
        variant=name, M=M, cols=COLS, K=K, fidelity=str(FID), fp32=FP32, qb=str(QB_DTYPE),
        ns_lo=res[NB_LO][0], ns_hi=res[NB_HI][0], per_block_ns=per_block, pcc=res[NB_HI][1],
        sum=float(res[NB_HI][2].double().sum()), host=socket.gethostname(), t=time.time(),
    )  # fmt: skip
    print("MM_RESULT", json.dumps(rec))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert res[NB_HI][1] > 0.999 and res[NB_LO][1] > 0.999, rec
