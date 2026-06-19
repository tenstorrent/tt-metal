# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

Two regimes (selected by the host heuristic):
  - Regime A: row-parallel. Each core owns a disjoint set of tile-rows and holds a
    full tile-row (Wt tiles) resident; zero cross-core communication.
  - Regime B: wide-W cross-core W-split. Each core owns a Wt/K shard of a row-group,
    computes its local partial sum-of-squares, all-gathers the K partials over an
    mcast rectangle, sums them, then normalizes its own shard.

The A-vs-B decision is by L1 fit: Regime A whenever a full row (input + gamma if
present) fits the resident budget; otherwise Regime B with K chosen to split W into
rectangular bands that saturate the grid.
"""

from pathlib import Path
import struct

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names live at the kernel boundary).
CB_INPUT_RESIDENT = 0
CB_GAMMA = 1
CB_SCALER = 8
CB_PARTIALS_GATHERED = 9
CB_OUTPUT = 16
CB_SQUARED = 24
CB_PARTIAL_SUMSQ = 25
CB_RECIP_RMS = 26
CB_NORMALIZED = 27

# Resident budget in tiles (≈1.12 MB for bf16), per op_design.md P3.
RESIDENT_BUDGET_TILES = 560


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _resolve_compute_config(compute_kernel_config):
    if compute_kernel_config is not None:
        return compute_kernel_config
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _dest_limit(cfg) -> int:
    fp32 = bool(getattr(cfg, "fp32_dest_acc_en", True))
    full_sync = bool(getattr(cfg, "dst_full_sync_en", False))
    if full_sync:
        return 8 if fp32 else 16
    return 4 if fp32 else 8


def _even_split(total, num_cores):
    """Return a list of (start, count) contiguous chunks summing to `total`."""
    base = total // num_cores
    rem = total % num_cores
    out = []
    start = 0
    for c in range(num_cores):
        count = base + (1 if c < rem else 0)
        out.append((start, count))
        start += count
    return out


def create_program_descriptor(input_tensor, output_tensor, gamma, epsilon, compute_kernel_config):
    has_gamma = gamma is not None

    cfg = _resolve_compute_config(compute_kernel_config)

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y

    padded = input_tensor.padded_shape
    Wt = padded[-1] // 32
    total_tiles = input_tensor.buffer_num_pages()
    Ht_total = total_tiles // Wt

    W = int(input_tensor.shape[-1])
    inv_W_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    resident_tiles = Wt + (Wt if has_gamma else 0)
    regime_a = resident_tiles <= RESIDENT_BUDGET_TILES

    if regime_a:
        return _regime_a_descriptor(
            input_tensor,
            output_tensor,
            gamma,
            has_gamma,
            cfg,
            Wt,
            Ht_total,
            total_cores,
            inv_W_bits,
            eps_bits,
        )
    return _regime_b_descriptor(
        input_tensor,
        output_tensor,
        gamma,
        has_gamma,
        cfg,
        Wt,
        Ht_total,
        grid,
        total_cores,
        inv_W_bits,
        eps_bits,
    )


def _regime_a_descriptor(
    input_tensor,
    output_tensor,
    gamma,
    has_gamma,
    cfg,
    Wt,
    Ht_total,
    total_cores,
    inv_W_bits,
    eps_bits,
):
    num_cores = min(Ht_total, total_cores)
    reduce_block = min(Wt, _dest_limit(cfg))

    tile_bytes = input_tensor.buffer_page_size()

    core_ranges = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, num_cores, row_wise=True)
    splits = _even_split(Ht_total, num_cores)

    # ---------- circular buffers ----------
    def cb(index, dtype, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_bytes,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
        )

    dt = input_tensor.dtype
    cbs = [
        cb(CB_INPUT_RESIDENT, dt, Wt),
        cb(CB_SCALER, dt, 1),
        cb(CB_OUTPUT, dt, 2),
        cb(CB_SQUARED, dt, reduce_block),
        cb(CB_PARTIAL_SUMSQ, dt, 2),
        cb(CB_RECIP_RMS, dt, 2),
    ]
    if has_gamma:
        cbs.append(cb(CB_GAMMA, dt, Wt))
        cbs.append(cb(CB_NORMALIZED, dt, Wt))

    # ---------- reader ----------
    reader_ct = [CB_INPUT_RESIDENT, CB_GAMMA, CB_SCALER, Wt, int(has_gamma)]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for core, (start, count) in zip(cores, splits):
        reader_rt[core.x][core.y] = [input_tensor.buffer_address(), gamma_addr, start, count]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, count]
        compute_rt[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer ----------
    writer_ct = [CB_OUTPUT, Wt]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute ----------
    compute_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        CB_PARTIALS_GATHERED,
        CB_OUTPUT,
        CB_SQUARED,
        CB_PARTIAL_SUMSQ,
        CB_RECIP_RMS,
        CB_NORMALIZED,
        Wt,
        reduce_block,
        int(has_gamma),
        inv_W_bits,
        eps_bits,
        1,  # num_partials = 1
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=cfg,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
    io = [input_tensor, gamma, output_tensor] if has_gamma else [input_tensor, output_tensor]
    return program, io


def _regime_b_descriptor(*args, **kwargs):
    raise NotImplementedError("rms_norm Regime B (cross-core W-split) is implemented in Stage 2.")
