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
# Regime B only: single-push handoff of the fully-accumulated local Sum(x^2) from
# compute (PASS-1) to the mcast reader. Decoupled from CB_PARTIAL_SUMSQ so the reader's
# cb_wait_front observes only the final value (Refinement 1 correctness fix).
CB_LOCAL_SUMSQ = 28

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
    row_fits = resident_tiles <= RESIDENT_BUDGET_TILES

    # Grid-aware heuristic (op_design.md "Grid-aware host heuristic"):
    #   - Regime A only when it already saturates the grid (Ht_total >= total_cores)
    #     AND a full row fits L1.
    #   - Otherwise prefer Regime B (W-split) when a rectangular partition exists that
    #     ADDS cores over what Regime A would use; else fall back to Regime A.
    if Ht_total >= total_cores and row_fits:
        return _regime_a_descriptor(
            input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, total_cores, inv_W_bits, eps_bits
        )

    K = _select_k(Wt, Ht_total, grid, total_cores, has_gamma)
    regime_a_cores = min(Ht_total, total_cores)
    if K is not None and Ht_total * K > regime_a_cores:
        return _regime_b_descriptor(
            input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, grid, total_cores, inv_W_bits, eps_bits
        )

    # Documented clean fallback: row-parallel on min(Ht_total, total_cores) cores.
    if not row_fits:
        raise NotImplementedError(
            f"rms_norm: row (Wt={Wt}, gamma={has_gamma}) exceeds L1 budget and no rectangular "
            f"Regime B partition exists for row_groups={Ht_total}, grid=({grid.x},{grid.y})."
        )
    return _regime_a_descriptor(
        input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, total_cores, inv_W_bits, eps_bits
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
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start * Wt, count * Wt]
        compute_rt[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer ----------
    writer_ct = [CB_OUTPUT]
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
        CB_PARTIAL_SUMSQ,  # cb_local_sumsq (unused in Regime A; num_partials==1 elides it)
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


def _select_k(Wt, num_row_groups, grid, total_cores, has_gamma):
    """Largest W-split factor K forming full-width rectangular bands that saturate the
    grid: K divides Wt, K is a multiple of grid.x (full-width bands), num_row_groups*K
    fits the grid, and Wt/K fits the resident budget. Returns None if none qualifies."""
    gx = grid.x
    budget = RESIDENT_BUDGET_TILES
    best = None
    for K in range(gx, total_cores + 1, gx):
        if Wt % K != 0:
            continue
        if K % gx != 0:
            continue
        if num_row_groups * K > total_cores:
            continue
        Wt_s = Wt // K
        if Wt_s + (Wt_s if has_gamma else 0) > budget:
            continue
        if best is None or K > best:
            best = K
    return best


def _regime_b_descriptor(
    input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, grid, total_cores, inv_W_bits, eps_bits
):
    num_row_groups = Ht_total  # Phase 0: bh = 1 tile-row per group
    gx = grid.x
    K = _select_k(Wt, num_row_groups, grid, total_cores, has_gamma)
    if K is None:
        raise NotImplementedError(
            f"rms_norm: no rectangular Regime B partition for Wt={Wt}, "
            f"row_groups={num_row_groups}, grid=({grid.x},{grid.y})"
        )

    gh = K // gx  # band height (rows) per group
    Wt_s = Wt // K
    reduce_block = min(Wt_s, _dest_limit(cfg))
    tile_bytes = input_tensor.buffer_page_size()
    dt = input_tensor.dtype
    device = input_tensor.device()

    DATA_READY = 0
    CONSUMED = 1

    # All cores used (num_row_groups bands of K cores each).
    used_cores = num_row_groups * K
    core_ranges = ttnn.num_cores_to_corerangeset(used_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    def cb(index, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_bytes,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dt, page_size=tile_bytes)],
        )

    cbs = [
        cb(CB_INPUT_RESIDENT, Wt_s),
        cb(CB_SCALER, 1),
        cb(CB_OUTPUT, 2),
        cb(CB_SQUARED, reduce_block),
        cb(CB_PARTIAL_SUMSQ, 2),
        cb(CB_RECIP_RMS, 2),
        cb(CB_PARTIALS_GATHERED, K),
        cb(CB_LOCAL_SUMSQ, 2),
    ]
    if has_gamma:
        cbs.append(cb(CB_GAMMA, Wt_s))
        cbs.append(cb(CB_NORMALIZED, Wt_s))

    # Semaphores on the full union of used cores (disjoint groups reuse the same IDs).
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=core_ranges, initial_value=0),
    ]

    def vcoord(lx, ly):
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
        return v.x, v.y

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for g in range(num_row_groups):
        # Group rectangle (logical): full grid width, rows [g*gh, g*gh+gh)
        rx0, ry0 = 0, g * gh
        rx1, ry1 = gx - 1, g * gh + gh - 1
        vrx0, vry0 = vcoord(rx0, ry0)
        vrx1, vry1 = vcoord(rx1, ry1)
        # Sender virtual coords for each rank j within this group.
        sender_coords = []
        for j in range(K):
            jx, jy = j % gx, g * gh + j // gx
            vx, vy = vcoord(jx, jy)
            sender_coords.extend([vx, vy])

        for r in range(K):
            lx, ly = r % gx, g * gh + r // gx
            input_page_base = g * Wt + r * Wt_s
            gamma_page_base = r * Wt_s

            reader_rt[lx][ly] = [
                r,
                input_tensor.buffer_address(),
                gamma_addr,
                input_page_base,
                gamma_page_base,
                vrx0,
                vry0,
                vrx1,
                vry1,
            ] + sender_coords
            writer_rt[lx][ly] = [output_tensor.buffer_address(), input_page_base, Wt_s]
            compute_rt[lx][ly] = [1]  # one tile-row group per core

    # ---------- reader (mcast all-gather) ----------
    reader_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        CB_LOCAL_SUMSQ,
        CB_PARTIALS_GATHERED,
        Wt_s,
        int(has_gamma),
        K,
        DATA_READY,
        CONSUMED,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader_mcast.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer ----------
    writer_ct = [CB_OUTPUT]
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
        Wt_s,
        reduce_block,
        int(has_gamma),
        inv_W_bits,
        eps_bits,
        K,
        CB_LOCAL_SUMSQ,
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
        semaphores=semaphores,
        cbs=cbs,
    )
    io = [input_tensor, gamma, output_tensor] if has_gamma else [input_tensor, output_tensor]
    return program, io
