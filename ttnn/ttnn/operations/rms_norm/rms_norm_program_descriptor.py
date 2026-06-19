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
import math
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
# Refinement 3 (TILE input + ROW_MAJOR gamma): the reader stages row-major gamma
# sticks here and compute tilizes them once into CB_GAMMA. Only allocated when
# gamma is supplied ROW_MAJOR with a TILE input.
CB_GAMMA_RM = 3

# Resident budget in tiles (≈1.12 MB for bf16), per op_design.md P3.
RESIDENT_BUDGET_TILES = 560

# ---- ROW_MAJOR (tilize-wrapped) regime CB indices ----
# Refinement 3: ROW_MAJOR input/output handled natively via a tilize-wrapped,
# row-parallel path. The math is identical to Regime A but reads/writes
# row-major sticks. Distinct from the TILE-regime CB map above.
CB_RM_IN = 0  # row-major input sticks (reader -> compute tilize)
CB_RM_GAMMA = 1  # row-major gamma sticks (reader -> compute tilize)
CB_RM_GAMMA_TILED = 2  # tilized gamma chunk (compute internal)
CB_RM_INPUT_RESIDENT = 24  # tilized resident input block (compute internal)
CB_RM_OUT = 16  # row-major output sticks (compute untilize -> writer)
CB_RM_SQUARED = 25
CB_RM_PARTIAL_SUMSQ = 26
CB_RM_RECIP_RMS = 27
CB_RM_NORMALIZED = 28
CB_RM_OUT_TILED = 29


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


def _intermediate_dtype(input_dtype, fp32_acc):
    """CB format for the accumulator/phase-boundary intermediates (Σx², the
    reduce scaler, recip-rms, the pass-2 normalized block, and the Regime-B
    gathered partials).

    Numeric-formats rule (skill §4): when the running Σx² crosses the CB,
    promote it to Float32 if fp32_dest_acc_en so the fp32 dest accumulation is
    not truncated at the pack boundary. A block-float (bf8b) input must never
    keep a bf8b accumulator — promote it. bf16 input keeps its bf16
    intermediates (byte-identical to Phase 0 / Refinement 1, no regression):
    bf16 ⊂ TF32 so the dest math is already exact and the baseline passes.
    """
    if input_dtype == ttnn.bfloat16:
        return ttnn.bfloat16
    # float32 or bfloat8_b input.
    return ttnn.float32 if fp32_acc else ttnn.bfloat16


def _tile_bytes(dtype) -> int:
    """Bytes for a standard 32x32 tile of `dtype` (Float32=4096, bf16=2048,
    bf8b=1088). Used for per-CB sizing now that input / intermediate / gamma /
    output CBs can each carry a different format."""
    return ttnn.tile_size(dtype)


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

    # Refinement 3: ROW_MAJOR input is handled by a dedicated tilize-wrapped,
    # row-parallel path (math stays on tiles). TILE input keeps the two-regime
    # (A / B) heuristic below, untouched.
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        return _regime_rm_descriptor(input_tensor, output_tensor, gamma, has_gamma, cfg, epsilon)

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

    core_ranges = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, num_cores, row_wise=True)
    splits = _even_split(Ht_total, num_cores)

    # ---------- circular buffers (per-CB format) ----------
    # Input / output CBs follow the tensor dtype; accumulator intermediates
    # (Σx², scaler, recip-rms, normalized block) follow _intermediate_dtype;
    # gamma follows its own dtype (mixed precision). Each format gets its own
    # tile byte size.
    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    # Refinement 3: TILE input may pair with ROW_MAJOR gamma. When so, the reader
    # stages gamma sticks in CB_GAMMA_RM and compute tilizes them into CB_GAMMA,
    # which must be padded to a whole number of reduce_block chunks.
    gamma_is_rm = 1 if (has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT) else 0
    num_chunks = (Wt + reduce_block - 1) // reduce_block
    W = int(input_tensor.shape[-1])
    # gamma_elem is only consumed by the reader's ROW_MAJOR-gamma path, where gamma
    # is guaranteed non-bf8b ({bf8b, ROW_MAJOR} is INVALID). Never call
    # element_size() on a block format (bf8b) — it raises "datum invalid".
    gamma_elem = gamma.element_size() if gamma_is_rm else 0
    cb_gamma_pages = (num_chunks * reduce_block) if gamma_is_rm else Wt
    cbs = [
        cb(CB_INPUT_RESIDENT, dt, Wt),
        cb(CB_SCALER, inter, 1),
        cb(CB_OUTPUT, dt, 2),
        cb(CB_SQUARED, inter, reduce_block),
        cb(CB_PARTIAL_SUMSQ, inter, 2),
        cb(CB_RECIP_RMS, inter, 2),
    ]
    if has_gamma:
        cbs.append(cb(CB_GAMMA, gamma_dt, cb_gamma_pages))
        # cb_normalized is the pass-2 Col->Row streaming intermediate, sized to one
        # REDUCE_BLOCK (constant) — NOT Wt — so the resident L1 footprint does not scale
        # with the row width (compute streams pass-2 per block).
        cbs.append(cb(CB_NORMALIZED, inter, reduce_block))
        if gamma_is_rm:
            cbs.append(cb(CB_GAMMA_RM, gamma_dt, 2 * reduce_block))

    # ---------- reader ----------
    reader_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        Wt,
        int(has_gamma),
        gamma_is_rm,
        CB_GAMMA_RM,
        reduce_block,
        num_chunks,
        W,
        gamma_elem,
    ]
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
        gamma_is_rm,
        CB_GAMMA_RM,
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
    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    device = input_tensor.device()
    # Refinement 3: TILE input + ROW_MAJOR gamma. Reader stages gamma sticks in
    # CB_GAMMA_RM; compute tilizes them into CB_GAMMA (padded to whole chunks).
    gamma_is_rm = 1 if (has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT) else 0
    num_chunks = (Wt_s + reduce_block - 1) // reduce_block
    W = int(input_tensor.shape[-1])
    # See Regime A note: never call element_size() on bf8b. gamma_elem is only used
    # by the reader's ROW_MAJOR-gamma path (gamma guaranteed non-bf8b there).
    gamma_elem = gamma.element_size() if gamma_is_rm else 0
    cb_gamma_pages = (num_chunks * reduce_block) if gamma_is_rm else Wt_s

    DATA_READY = 0
    CONSUMED = 1

    # All cores used (num_row_groups bands of K cores each).
    used_cores = num_row_groups * K
    core_ranges = ttnn.num_cores_to_corerangeset(used_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    # Per-CB format: input/output follow the tensor dtype; the mcast-gathered
    # partials (cb_local_sumsq → cb_partials_gathered) and the accumulator
    # intermediates follow _intermediate_dtype, so the cross-core all-gather
    # transfers a matched tile-byte count (both endpoints share the format).
    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    cbs = [
        cb(CB_INPUT_RESIDENT, dt, Wt_s),
        cb(CB_SCALER, inter, 1),
        cb(CB_OUTPUT, dt, 2),
        cb(CB_SQUARED, inter, reduce_block),
        cb(CB_PARTIAL_SUMSQ, inter, 2),
        cb(CB_RECIP_RMS, inter, 2),
        cb(CB_PARTIALS_GATHERED, inter, K),
        cb(CB_LOCAL_SUMSQ, inter, 2),
    ]
    if has_gamma:
        cbs.append(cb(CB_GAMMA, gamma_dt, cb_gamma_pages))
        # Constant-sized pass-2 streaming intermediate (one REDUCE_BLOCK), not Wt_s.
        cbs.append(cb(CB_NORMALIZED, inter, reduce_block))
        if gamma_is_rm:
            cbs.append(cb(CB_GAMMA_RM, gamma_dt, 2 * reduce_block))

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
        gamma_is_rm,
        CB_GAMMA_RM,
        reduce_block,
        num_chunks,
        W,
        gamma_elem,
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
        gamma_is_rm,
        CB_GAMMA_RM,
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


def _regime_rm_descriptor(input_tensor, output_tensor, gamma, has_gamma, cfg, epsilon):
    """ROW_MAJOR (tilize-wrapped) row-parallel descriptor.

    Each stick (one (b,c,h) row of W elements) is RMS-normalized independently.
    Sticks are processed in 32-stick tile-blocks; the W axis is chunked by
    `reduce_block` tiles so the per-core L1 footprint is bounded regardless of W.
    Non-tile-aligned W (zero-padded columns) and H (partial last block) are
    handled natively in the dataflow kernels — no host-side pad/slice/to_layout.
    """
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y

    shape = list(input_tensor.shape)
    W = int(shape[-1])
    total_sticks = 1
    for d in shape[:-1]:
        total_sticks *= int(d)

    Wt = (W + 31) // 32
    num_blocks_total = (total_sticks + 31) // 32  # 32 sticks per tile-block

    reduce_block = min(Wt, _dest_limit(cfg))
    num_chunks = (Wt + reduce_block - 1) // reduce_block
    Wt_padded = num_chunks * reduce_block

    inv_W_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    in_elem = input_tensor.element_size()  # RM input is never bf8b ({bf8b, RM} INVALID)
    out_elem = output_tensor.element_size()
    gamma_is_tile = 1 if (has_gamma and gamma.layout == ttnn.TILE_LAYOUT) else 0
    # gamma_elem only feeds the reader's ROW_MAJOR-gamma path (gamma non-bf8b there;
    # bf8b gamma is always TILE). Avoid element_size() on a bf8b TILE gamma.
    gamma_elem = gamma.element_size() if (has_gamma and not gamma_is_tile) else in_elem

    num_cores = min(num_blocks_total, total_cores)
    core_ranges = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, num_cores, row_wise=True)
    splits = _even_split(num_blocks_total, num_cores)

    db = 2  # double-buffer for streamed chunk CBs

    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    cbs = [
        cb(CB_RM_IN, dt, db * reduce_block),
        cb(CB_SCALER, inter, 1),
        cb(CB_RM_OUT, dt, db * reduce_block),
        cb(CB_RM_INPUT_RESIDENT, dt, Wt_padded),
        cb(CB_RM_SQUARED, inter, reduce_block),
        cb(CB_RM_PARTIAL_SUMSQ, inter, 2),
        cb(CB_RM_RECIP_RMS, inter, 2),
        cb(CB_RM_OUT_TILED, dt, db * reduce_block),
    ]
    if has_gamma:
        cbs.append(cb(CB_RM_GAMMA, gamma_dt, db * reduce_block))
        cbs.append(cb(CB_RM_GAMMA_TILED, gamma_dt, db * reduce_block))
        cbs.append(cb(CB_RM_NORMALIZED, inter, reduce_block))

    # ---------- reader ----------
    reader_ct = [
        CB_RM_IN,
        CB_RM_GAMMA,
        CB_RM_GAMMA_TILED,
        CB_SCALER,
        int(has_gamma),
        gamma_is_tile,
        Wt,
        reduce_block,
        num_chunks,
        W,
        in_elem,
        gamma_elem,
    ]
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
        reader_rt[core.x][core.y] = [input_tensor.buffer_address(), gamma_addr, start, count, total_sticks]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, count, total_sticks]
        compute_rt[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader_rm.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer ----------
    writer_ct = [CB_RM_OUT, Wt, reduce_block, num_chunks, W, out_elem]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer_rm.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute ----------
    compute_ct = [
        CB_RM_IN,
        CB_RM_GAMMA,
        CB_RM_GAMMA_TILED,
        CB_SCALER,
        CB_RM_OUT,
        CB_RM_INPUT_RESIDENT,
        CB_RM_SQUARED,
        CB_RM_PARTIAL_SUMSQ,
        CB_RM_RECIP_RMS,
        CB_RM_NORMALIZED,
        CB_RM_OUT_TILED,
        reduce_block,
        num_chunks,
        int(has_gamma),
        gamma_is_tile,
        inv_W_bits,
        eps_bits,
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute_rm.cpp"),
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
