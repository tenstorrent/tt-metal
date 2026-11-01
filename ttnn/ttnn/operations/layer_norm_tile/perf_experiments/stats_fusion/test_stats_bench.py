# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated variance-stage bench (one core): center -> sumsq -> inv_std [-> normalize] on resident z + mean.

    scripts/run_safe_pytest.sh --run-all [--profile] <this file> --import-mode=prepend -s

Variant = STATS_VARIANT of stats_variants.hpp (0 baseline, 1 tail_fold, 2 tail_fused, 3 fused_only, 4 tail_col0, 5 col0_only).
mode 0 = timed pipeline incl. normalize (the real consumer of inv_std); mode 1 = writes c and inv_std
so inv_std precision is measured directly. Correctness gate vs float64 torch on every run.
"""

import math
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

KDIR = Path(__file__).parent / "bench_kernels"
CB_Z, CB_MEAN, CB_SCALER, CB_WORK_B, CB_SUMSQ, CB_INV_STD, CB_OUT = 0, 1, 2, 3, 4, 5, 6
ITERATIONS = 24
EPS = 1e-5


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _run(device, variant, mode, Wt, fp32_dest, offset, diag=False):
    W = Wt * 32
    torch.manual_seed(0)
    scale = torch.linspace(0.5, 3.0, 32).reshape(32, 1)
    z = (torch.randn(32, W) * scale + offset).float()
    if not fp32_dest:
        z = z.to(torch.bfloat16).float()
    mean = z.double().mean(dim=-1, keepdim=True).float()
    mean_tile = mean.expand(32, 32).contiguous()

    work_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16
    tile_bytes = ttnn.tile_size(work_dtype)
    mk = lambda t: ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=work_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    z_t, mean_t = mk(z), mk(mean_tile)
    main_t = mk(torch.zeros(32, W))
    inv_t = mk(torch.zeros(32, 32))

    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    pages = {CB_Z: Wt, CB_MEAN: 1, CB_SCALER: 1, CB_WORK_B: Wt, CB_SUMSQ: 1, CB_INV_STD: 1, CB_OUT: Wt}
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * tile_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=work_dtype, page_size=tile_bytes)],
        )
        for i, n in pages.items()
    ]

    reader_ct = [CB_Z, CB_MEAN, Wt, ITERATIONS, tile_bytes]
    reader_ct += ttnn.TensorAccessorArgs(z_t).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(mean_t).get_compile_time_args()
    writer_ct = [CB_OUT if mode == 0 else CB_WORK_B, CB_INV_STD, Wt, ITERATIONS, tile_bytes, mode]
    writer_ct += ttnn.TensorAccessorArgs(main_t).get_compile_time_args()
    writer_ct += ttnn.TensorAccessorArgs(inv_t).get_compile_time_args()
    compute_ct = [
        CB_Z,
        CB_MEAN,
        CB_SCALER,
        CB_WORK_B,
        CB_SUMSQ,
        CB_INV_STD,
        CB_OUT,
        Wt,
        ITERATIONS,
        _bits(1.0 / W),
        _bits(EPS),
        4,
        mode,
    ]

    def rt(vals):
        r = ttnn.RuntimeArgs()
        r[core.x][core.y] = vals
        return r

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=rt([z_t.buffer_address(), mean_t.buffer_address()]),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=rt([main_t.buffer_address(), inv_t.buffer_address()]),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            defines=[("STATS_VARIANT", str(variant))] + ([("STATS_DIAG", "1")] if diag else []),
            runtime_args=rt([]),
            config=cfg,
        ),
    ]
    ttnn.generic_op([z_t, mean_t, main_t, inv_t], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    c_ref = z.double() - mean.double()
    inv_ref = 1.0 / torch.sqrt((c_ref * c_ref).mean(dim=-1) + EPS)
    main = ttnn.to_torch(main_t).reshape(32, W).double()
    if mode == 0:
        ref = c_ref * inv_ref.reshape(32, 1)
        rel_rms = math.sqrt(((main - ref) ** 2).mean().item()) / ref.std().item()
        print(f"\nBENCH v={variant} mode=0 Wt={Wt} fp32={fp32_dest} off={offset} y_rel_rms={rel_rms:.3e}")
        assert rel_rms < 0.02
    else:
        inv = ttnn.to_torch(inv_t).reshape(32, 32)[:, 0].double()
        inv_rel = (inv - inv_ref).abs() / inv_ref
        inv_rms = math.sqrt((inv_rel**2).mean().item())
        print(
            f"\nBENCH v={variant} mode=1 Wt={Wt} fp32={fp32_dest} off={offset} "
            f"inv_std_rel_rms={inv_rms:.3e} inv_std_rel_max={inv_rel.max().item():.3e}"
        )
        assert inv_rms < 0.01


VARIANTS = [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("variant", VARIANTS)
def test_timed(device, variant):
    _run(device, variant, 0, 24, True, 0.0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("offset", [0.0, 100.0], ids=["zero_mean", "mean100"])
def test_inv_std_precision(device, variant, fp32_dest, offset):
    _run(device, variant, 1, 24, fp32_dest, offset)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("Wt", [4, 12, 48])
def test_timed_width(device, variant, Wt):
    _run(device, variant, 0, Wt, True, 0.0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_timed_bf16dest(device, variant):
    _run(device, variant, 0, 24, False, 0.0)


@pytest.mark.parametrize("variant", [2, 4])
def test_diag(device, variant):
    _run(device, variant, 0, 24, True, 0.0, diag=True)
