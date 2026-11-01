# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated post-stats chain bench (one core): center -> row_stats -> normalize -> scale -> shift on resident
fp32 z (= x + res, the op's residual path) + fp32 mean, bf16 gamma/beta tiles, bf16 output.

    scripts/run_safe_pytest.sh --run-all [--profile] <this file>::<test> --import-mode=prepend -s

CHAIN_VARIANT (env PSC_VARIANTS="0,1,..") selects the variant of chain_variants.hpp. Correctness vs float64
torch on every run (PCC 0.999 / rel-RMS 0.02 golden gate); metrics printed as BENCH lines.
"""

import os
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

from eval.metrics import compute_metrics_torch

KDIR = Path(__file__).parent / "bench_kernels"
CB_Z, CB_MEAN, CB_GAMMA, CB_BETA, CB_WORK_A, CB_WORK_B, CB_INV_STD, CB_AUX, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 7, 16
ITERATIONS = int(os.environ.get("PSC_ITERS", "16"))
EPS = 1e-5
VARIANTS = [int(v) for v in os.environ.get("PSC_VARIANTS", "0").split(",")]


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def run_chain(
    device, variant, *, rows=1, Wt=24, fp32_dest=True, offset=0.0, lanes=4, diag=False, aux_fp32=True, work_bf16=False
):
    W = Wt * 32
    H = rows * 32
    torch.manual_seed(0)
    x = (torch.randn(H, W) * 2.0 + offset).to(torch.bfloat16)
    res = torch.randn(H, W).to(torch.bfloat16)
    z = x.float() + res.float()  # the op's fp32 work page (bf16 + bf16 is exact in fp32 here)
    gamma = torch.randn(W).to(torch.bfloat16)
    beta = torch.randn(W).to(torch.bfloat16)
    if not fp32_dest:
        z = z.to(torch.bfloat16).float()
    mean = z.double().mean(dim=-1, keepdim=True)
    expected = torch.nn.functional.layer_norm(z.double(), (W,), gamma.double(), beta.double(), eps=EPS)

    work_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16
    work_bytes = ttnn.tile_size(work_dtype)
    io_bytes = ttnn.tile_size(ttnn.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG
    mk = lambda t, dt: ttnn.from_torch(
        t.reshape(1, 1, *t.shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
    )
    # tile-row r of z is tiles [r * Wt, (r + 1) * Wt): row-major tile order of an (H, W) tensor
    z_t = mk(z, work_dtype)
    mean_t = mk(mean.float().expand(H, 32).contiguous(), work_dtype)
    gamma_t = mk(gamma.float().reshape(1, W).expand(32, W).contiguous(), ttnn.bfloat16)
    beta_t = mk(beta.float().reshape(1, W).expand(32, W).contiguous(), ttnn.bfloat16)
    out_t = mk(torch.zeros(H, W), ttnn.bfloat16)

    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    rt_ = rows * Wt
    aux_dtype = ttnn.float32 if aux_fp32 and fp32_dest else ttnn.bfloat16
    pages = {
        CB_Z: (rt_, work_dtype),
        CB_MEAN: (rows, work_dtype),
        CB_GAMMA: (Wt, ttnn.bfloat16),
        CB_BETA: (Wt, ttnn.bfloat16),
        CB_WORK_A: (rt_, ttnn.bfloat16 if work_bf16 else work_dtype),
        CB_WORK_B: (rt_, ttnn.bfloat16 if work_bf16 else work_dtype),
        CB_INV_STD: (rows, work_dtype),
        CB_AUX: (rt_, aux_dtype),
        CB_OUT: (rt_, ttnn.bfloat16),
    }
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * ttnn.tile_size(dt),
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=ttnn.tile_size(dt))],
        )
        for i, (n, dt) in pages.items()
    ]

    reader_ct = [CB_Z, CB_MEAN, CB_GAMMA, CB_BETA, rt_, rows, Wt, ITERATIONS, work_bytes, io_bytes]
    for t in (z_t, mean_t, gamma_t, beta_t):
        reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    writer_ct = [CB_OUT, rt_, ITERATIONS, io_bytes] + ttnn.TensorAccessorArgs(out_t).get_compile_time_args()
    compute_ct = [
        CB_Z,
        CB_MEAN,
        CB_GAMMA,
        CB_BETA,
        CB_WORK_A,
        CB_WORK_B,
        CB_INV_STD,
        CB_OUT,
        rows,
        Wt,
        ITERATIONS,
        _bits(1.0 / W),
        _bits(EPS),
        lanes,
        CB_AUX,
    ]

    def rt(vals):
        r = ttnn.RuntimeArgs()
        r[core.x][core.y] = vals
        return r

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    defines = [("CHAIN_VARIANT", str(variant))] + ([("CHAIN_DIAG", "1")] if diag else [])
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=rt(
                [z_t.buffer_address(), mean_t.buffer_address(), gamma_t.buffer_address(), beta_t.buffer_address()]
            ),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=rt([out_t.buffer_address()]),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            defines=defines,
            runtime_args=rt([]),
            config=cfg,
        ),
    ]
    ttnn.generic_op(
        [z_t, mean_t, gamma_t, beta_t, out_t], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    )
    actual = ttnn.to_torch(out_t).reshape(H, W).float()
    m = compute_metrics_torch(actual, expected.float(), readback_dtype=torch.bfloat16)
    print(
        f"\nBENCH v={variant} rows={rows} Wt={Wt} fp32={fp32_dest} off={offset} lanes={lanes} "
        f"work_bf16={work_bf16} pcc={m.pcc:.6f} rel_rms={m.rms:.6f}"
    )
    assert m.pcc >= 0.999 and m.rms <= 0.02, (m.pcc, m.rms)
    return m


@pytest.mark.parametrize("variant", VARIANTS)
def test_timed(device, variant):
    run_chain(device, variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_diag(device, variant):
    run_chain(device, variant, diag=True)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows", [2])
def test_rows2(device, variant, rows):
    run_chain(device, variant, rows=rows)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("Wt", [1, 3, 8, 48])
def test_width(device, variant, Wt):
    run_chain(device, variant, Wt=Wt)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bf16dest(device, variant):
    run_chain(device, variant, fp32_dest=False)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("offset", [0.0, 100.0], ids=["zero_mean", "mean100"])
def test_precision(device, variant, fp32_dest, offset):
    run_chain(device, variant, fp32_dest=fp32_dest, offset=offset)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("offset", [0.0, 100.0], ids=["zero_mean", "mean100"])
def test_work_bf16(device, variant, offset):
    """Diagnostic / precision-trade option: work_a / work_b (c, n, n * gamma) in bf16 pages."""
    run_chain(device, variant, work_bf16=True, offset=offset, diag=offset == 0.0)
