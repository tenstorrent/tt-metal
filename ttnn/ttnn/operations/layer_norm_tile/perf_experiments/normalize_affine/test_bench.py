# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated normalize/affine stage bench (one core, c / inv_std / gamma / beta pre-resident).

scripts/run_safe_pytest.sh [--profile] <this file> --import-mode=prepend [-k ...]
Per-rep cost = (DEVICE KERNEL DURATION[reps=33] - [reps=1]) / 32. Correctness: rep 0 vs torch.
NA_BENCH_ORDER=reverse flips the variant order (alternate A/B, B/A between runs).
"""

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from install import DIAG, VARIANTS  # noqa: E402

KERNELS = Path(__file__).parent / "kernels"
TILE = 32
CB_C_SRC, CB_INV_SRC, CB_GAMMA, CB_BETA, CB_WA, CB_WB, CB_INV, CB_OUT = 0, 1, 2, 3, 4, 5, 8, 16
DEST_LANES = 4


def _cb(core_ranges, pages, members):
    page_size = ttnn.tile_size(members[0][1])
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size) for idx, fmt in members
        ],
    )


def run_bench(device, variant, *, rows, Wt, reps, fp32_dest, fidelity=ttnn.MathFidelity.HiFi4):
    torch.manual_seed(0)
    work = ttnn.float32 if fp32_dest else ttnn.bfloat16
    # Values bf16-representable so the staging copy through srcA is exact in both DEST modes.
    c = torch.randn(rows * TILE, Wt * TILE).to(torch.bfloat16).float()
    inv = (torch.rand(rows * TILE, TILE) * 2 + 0.25).to(torch.bfloat16).float()  # only column 0 is meaningful
    gamma = torch.randn(TILE, Wt * TILE).to(torch.bfloat16)  # only row 0 is meaningful
    beta = torch.randn(TILE, Wt * TILE).to(torch.bfloat16)
    expected = c * inv[:, :1] * gamma[0].float() + beta[0].float()

    dram = ttnn.DRAM_MEMORY_CONFIG
    t_c = ttnn.from_torch(c, dtype=work, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    t_inv = ttnn.from_torch(inv, dtype=work, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    t_g = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    t_b = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    t_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows * TILE, Wt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    n = rows * Wt
    cbs = [
        _cb(cores, n, [(CB_C_SRC, work)]),
        _cb(cores, rows, [(CB_INV_SRC, work)]),
        _cb(cores, Wt, [(CB_GAMMA, ttnn.bfloat16)]),
        _cb(cores, Wt, [(CB_BETA, ttnn.bfloat16)]),
        _cb(cores, n, [(CB_WA, work), (CB_WB, work)]),
        _cb(cores, rows, [(CB_INV, work)]),
        _cb(cores, n, [(CB_OUT, ttnn.bfloat16)]),
    ]
    work_bytes = ttnn.tile_size(work)
    io_bytes = ttnn.tile_size(ttnn.bfloat16)

    reader_ct = [CB_C_SRC, CB_INV_SRC, CB_GAMMA, CB_BETA, rows, Wt, work_bytes, io_bytes]
    for t in (t_c, t_inv, t_g, t_b):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [CB_OUT, n, reps, io_bytes] + list(ttnn.TensorAccessorArgs(t_out).get_compile_time_args())
    compute_ct = [
        CB_C_SRC,
        CB_INV_SRC,
        CB_GAMMA,
        CB_BETA,
        CB_WA,
        CB_WB,
        CB_INV,
        CB_OUT,
        rows,
        Wt,
        reps,
        {**VARIANTS, **DIAG}[variant],
        DEST_LANES,
    ]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_rt[0][0] = [t.buffer_address() for t in (t_c, t_inv, t_g, t_b)]
    writer_rt[0][0] = [t_out.buffer_address()]
    compute_rt[0][0] = []

    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = fidelity
    config.fp32_dest_acc_en = fp32_dest
    config.math_approx_mode = False

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bench_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bench_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bench_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=config,
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([t_c, t_inv, t_g, t_b, t_out], program)
    actual = ttnn.to_torch(out).float()
    exp = expected.to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([actual.flatten(), exp.flatten()]))[0, 1].item()
    err = actual - expected
    rel_rms = (err.pow(2).mean().sqrt() / expected.std()).item()
    print(
        f"NA_BENCH variant={variant} rows={rows} Wt={Wt} reps={reps} fp32_dest={fp32_dest} fid={str(fidelity).split('.')[-1]} "
        f"pcc={pcc:.7f} rel_rms={rel_rms:.6f}"
    )
    if variant in DIAG:
        return
    assert pcc >= (0.999 if fp32_dest else 0.998) and rel_rms <= (0.02 if fp32_dest else 0.05)


ORDER = ["baseline", "parked", "parked_lean", "seed", "seed_raw", "seed_il", "seed_pipe"]
if os.environ.get("NA_BENCH_ORDER") == "reverse":
    ORDER = ORDER[::-1]
SHAPES = [
    (1, 24, True),  # focus: R=1, Wt=24, fp32 DEST
    (1, 24, False),
    (2, 24, True),
    (1, 8, True),
    (1, 64, True),
]


@pytest.mark.parametrize("reps", [1, 33])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"R{r}_Wt{w}_{'fp32' if f else 'bf16'}" for r, w, f in SHAPES])
@pytest.mark.parametrize("variant", ORDER)
def test_bench(device, variant, shape, reps):
    rows, Wt, fp32_dest = shape
    run_bench(device, variant, rows=rows, Wt=Wt, reps=reps, fp32_dest=fp32_dest)


@pytest.mark.parametrize("reps", [1, 33])
@pytest.mark.parametrize("fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi], ids=["HiFi4", "LoFi"])
@pytest.mark.parametrize("variant", list(DIAG))
def test_diag(device, variant, fidelity, reps):
    """Per-op cost diagnosis only (LoFi rows are NOT candidates: fidelity is the user's contract)."""
    run_bench(device, variant, rows=1, Wt=24, reps=reps, fp32_dest=True, fidelity=fidelity)
