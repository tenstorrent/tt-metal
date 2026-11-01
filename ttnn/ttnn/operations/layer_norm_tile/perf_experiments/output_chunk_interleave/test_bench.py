# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated output-stage bench (one core, c / inv_std / gamma / beta pre-resident, op-style chunked writer).

scripts/run_safe_pytest.sh [--profile] <this file> --import-mode=prepend [-k ...]
reps=1 kernel ns = output-stage latency incl. the write tail; (reps=33 - reps=1) / 32 = per-row compute cost.
OCI_ORDER=reverse flips the variant order. Correctness: rep 0 vs torch, and bit-exact vs base.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

KERNELS = Path(__file__).parent / "bench_kernels"
TILE = 32
CB_C_SRC, CB_INV_SRC, CB_GAMMA, CB_BETA, CB_WA, CB_WB, CB_WC, CB_INV, CB_OUT = 0, 1, 2, 3, 4, 5, 10, 8, 16
DEST_LANES = 4
VARIANTS = {
    "base": (0, 4),
    "helper_g8": (1, 8),
    "raw_g4": (2, 4),
    "raw_g8": (2, 8),
    "raw_g12": (2, 12),
    "rawskew_g4": (3, 4),
    "rawskew_g8": (3, 8),
    "helperskew_g4": (4, 4),
}
_BASE_OUT = {}


def _cb(core_ranges, pages, members):
    page_size = ttnn.tile_size(members[0][1])
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size) for idx, fmt in members
        ],
    )


def io_chunk(Wt):
    return max(c for c in range(1, min(Wt, 8) + 1) if Wt % c == 0)


def run_bench(device, variant, *, rows, Wt, reps, fp32_dest):
    torch.manual_seed(0)
    vid, chunk = VARIANTS[variant]
    work = ttnn.float32 if fp32_dest else ttnn.bfloat16
    c = torch.randn(rows * TILE, Wt * TILE).to(torch.bfloat16).float()
    inv = (torch.rand(rows * TILE, TILE) * 2 + 0.25).to(torch.bfloat16).float()
    gamma = torch.randn(TILE, Wt * TILE).to(torch.bfloat16)
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
        _cb(cores, 2 * chunk, [(CB_WC, work)]),
        _cb(cores, rows, [(CB_INV, work)]),
        _cb(cores, n, [(CB_OUT, ttnn.bfloat16)]),
    ]
    work_bytes = ttnn.tile_size(work)
    io_bytes = ttnn.tile_size(ttnn.bfloat16)

    reader_ct = [CB_C_SRC, CB_INV_SRC, CB_GAMMA, CB_BETA, rows, Wt, work_bytes, io_bytes]
    for t in (t_c, t_inv, t_g, t_b):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [CB_OUT, n, reps, io_bytes, io_chunk(Wt)] + list(ttnn.TensorAccessorArgs(t_out).get_compile_time_args())
    compute_ct = [CB_C_SRC, CB_INV_SRC, CB_GAMMA, CB_BETA, CB_WA, CB_WB, CB_WC, CB_INV, CB_OUT]
    compute_ct += [rows, Wt, reps, vid, DEST_LANES, chunk]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_rt[0][0] = [t.buffer_address() for t in (t_c, t_inv, t_g, t_b)]
    writer_rt[0][0] = [t_out.buffer_address()]
    compute_rt[0][0] = []

    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
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
    rel_rms = ((actual - expected).pow(2).mean().sqrt() / expected.std()).item()
    key = (rows, Wt, fp32_dest)
    if variant == "base":
        _BASE_OUT[key] = actual
    diff = (actual - _BASE_OUT[key]).abs().max().item() if key in _BASE_OUT else float("nan")
    print(
        f"OCI_BENCH variant={variant} rows={rows} Wt={Wt} reps={reps} fp32_dest={fp32_dest} "
        f"pcc={pcc:.7f} rel_rms={rel_rms:.6f} maxdiff_vs_base={diff}"
    )
    assert pcc >= (0.999 if fp32_dest else 0.998) and rel_rms <= (0.02 if fp32_dest else 0.05)
    if key in _BASE_OUT:
        assert diff == 0.0


ORDER = list(VARIANTS)
if os.environ.get("OCI_ORDER") == "reverse":
    ORDER = ORDER[::-1]
SHAPES = [
    (1, 24, True),  # focus: R=1, Wt=24, fp32 DEST
    (1, 24, False),
    (2, 24, True),
    (1, 12, True),
    (1, 64, True),
]


@pytest.mark.parametrize("reps", [1, 33])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"R{r}_Wt{w}_{'fp32' if f else 'bf16'}" for r, w, f in SHAPES])
@pytest.mark.parametrize("variant", ORDER)
def test_bench(device, variant, shape, reps):
    rows, Wt, fp32_dest = shape
    run_bench(device, variant, rows=rows, Wt=Wt, reps=reps, fp32_dest=fp32_dest)
