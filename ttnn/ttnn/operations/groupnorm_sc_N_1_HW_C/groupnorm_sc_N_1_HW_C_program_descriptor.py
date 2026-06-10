# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for groupnorm_sc_N_1_HW_C.

Multi-core: N*G (n, g) groups are split across the full compute grid via
ttnn.split_work_to_cores (work unit = one group; embarrassingly parallel over
interleaved DRAM — group slabs are disjoint, gamma/beta read by each core).
Per group the compute kernel makes three streaming passes over the Ht x Wg
tile slab (mean, variance, normalize+affine); the reader re-reads the slab
from DRAM each pass so HW can be arbitrarily large.

CB layout (per op_design.md, refined dtype handling):
  cb_input_tiles  (0):  2*Wg pages, in_dtype — pass 1/2/3 stream
  cb_gamma_tiles  (1):  Wt pages, affine dtype — read once, HeldBulk
  cb_beta_tiles   (2):  Wt pages, affine dtype — read once, HeldBulk
  cb_scaler       (8):  1 page, bf16 — 1/sqrt(HW*Cg) on LOGICAL sizes, pushed once
  cb_mask_interior(9):  Wg pages, bf16 — C-tail col mask row (c_tail > 0 only)
  cb_mask_tail    (10): Wg pages, bf16 — HW-tail row mask row (hw_tail > 0 only)
  cb_mean_export  (11): 2 pages, stat fmt — compute -> reader per-group mean scalar
  cb_rstd_export  (12): 2 pages, stat fmt — compute -> reader per-group rstd scalar
  cb_mean_row     (13): Wc pages, stat fmt — reader-built per-column mean row vector
  cb_rstd_row     (14): Wc pages, stat fmt — reader-built per-column rstd row vector
  cb_output_tiles (16): 2*Wg pages, out_dtype — compute -> writer stream
  cb_mean         (24): 1 page, stat fmt — per-group mean, HeldBulk passes 2/3
  cb_var          (25): 1 page, stat fmt — variance accumulator -> rstd
  cb_centered     (26): 2*Wg pages, stat fmt — (x - mean) per row chunk
  cb_xhat         (27): 2*Wg pages, stat fmt — (x - mean)*rstd (HAS_GAMMA only)
  cb_scaled       (28): 2*Wg pages, stat fmt — xhat*gamma (HAS_GAMMA && HAS_BETA)

Non-tile-aligned group widths (Refinement 3, Cg % 32 != 0, G > 1 — the SD /
SDXL regime): groups straddle tile boundaries, so the work unit changes from
one (n, g) group to one (n, cluster), where a cluster is lcm(Cg, 32) channels
(capped at C) — the smallest channel run on which group and tile boundaries
re-align. Group output tiles within a cluster stay shared, but cluster output
tiles are disjoint, keeping multi-core writes race-free. Per cluster:
  - Passes 1/2 run per group over its tile span (Wsmax <= ceil(Cg/32)+1 tiles)
    with per-group 0/1 column masks (cb_mask_interior / cb_mask_tail).
  - Compute exports each group's mean/rstd scalars (cb_mean_export /
    cb_rstd_export); the reader scatters them into per-column row vectors
    (cb_mean_row / cb_rstd_row), zeros in the padding columns.
  - Pass 3 is one Row-broadcast sweep over the whole cluster:
    (x - mean_row) * rstd_row * gamma + beta — no partial tile writes;
    padding columns get rstd=0 -> 0 output.
CB-wrap rule (deadlock otherwise): multi-tile reserve/push frames must have
uniform width per CB. Input/output stream per tile; mask frames pad to Ws_max
(2*Ws_max pages); row vectors push Wc_full frames (Wc_full pages); the bf8b
output row masks are single scalar tiles (cb_mask_ones 15 / cb_mask_rows 17).
Passes 1/2 accumulate in chunk_rows(=16)-row blocks: per-block partial reload
quantizes through TF32 srcA, so fewer/taller blocks keep stats accurate at
large Ht; only cb_centered is sized for the full chunk.

Intermediate (stat) format: Float32 when fp32_dest_acc_en is set (accumulation
crosses these CBs between passes — packing to bf16 would erase the fp32 dest
gain) or when the input itself is fp32; bfloat16 otherwise (incl. bf8b input —
block-float intermediates would lose precision for no L1 win). No CB carries
UnpackToDestMode::UnpackToDestFp32: every intermediate feeds FPU helpers
(sub/mul/square/reduce), which the tag forbids.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor,  # TILE-layout (1,1,1,C) or None
    beta_tensor,  # TILE-layout (1,1,1,C) or None
    output_tensor: ttnn.Tensor,
    num_groups: int,
    eps: float,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    N, _, HW, C = list(input_tensor.shape)
    G = num_groups
    Cg = C // G
    # Padded tile counts (HW / C tails round up).
    Ht = (HW + TILE_DIM - 1) // TILE_DIM
    Wt = (C + TILE_DIM - 1) // TILE_DIM
    Wg = (Cg + TILE_DIM - 1) // TILE_DIM
    # Tail sizes (0 = aligned).
    hw_tail = HW % TILE_DIM
    c_tail = C % TILE_DIM

    # Non-tile-aligned group widths (Refinement 3): groups straddle tile
    # boundaries. Work unit becomes one (n, cluster); cluster = lcm(Cg, 32)
    # channels (capped at C) so group and tile boundaries re-align at cluster
    # edges and cluster output tiles are disjoint across cores.
    groups_non_aligned = G > 1 and Cg % TILE_DIM != 0
    cluster_ch = (Cg * TILE_DIM) // math.gcd(Cg, TILE_DIM)
    num_clusters = (C + cluster_ch - 1) // cluster_ch
    Wc_full = cluster_ch // TILE_DIM  # cluster_t0 stride (full clusters)
    Wc_max = min(Wc_full, Wt)  # widest cluster in tiles (last may be capped)
    # Widest per-group tile span: a group covers ceil(Cg/32) tiles + 1 for
    # straddling a tile boundary, never wider than the cluster itself.
    Ws_max = min(Cg // TILE_DIM + 2, Wc_max)
    # Accumulation chunk (rows per reduce block) for cluster passes 1/2. Each
    # block reloads the running partial through TF32 srcA (truncation toward
    # zero): 512 single-row reloads at HW=16384 cost ~9% of the variance —
    # 16-row chunks cut that to ~0.3%. Bounded by L1 (R*Ws_max stat pages).
    chunk_rows = min(Ht, 16)

    has_gamma = gamma_tensor is not None
    has_beta = beta_tensor is not None

    # --- compute config ---
    # Defaults reproduce Phase-0 ComputeConfigDescriptor() for bf16/bf8b inputs.
    # fp32 input defaults fp32_dest_acc_en=True (dtype-driven; fp32 was never
    # supported before so there is no prior behavior to preserve, and TF32
    # dest rounding otherwise dominates the fp32 stat error budget — measured
    # rel_rms 0.0112 -> 0.0075 on (1,1,1024,256) G=8 + bf8b gamma).
    if compute_kernel_config is not None:
        math_fidelity = compute_kernel_config.math_fidelity
        fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en
        math_approx_mode = compute_kernel_config.math_approx_mode
        dst_full_sync_en = compute_kernel_config.dst_full_sync_en
    else:
        math_fidelity = ttnn.MathFidelity.HiFi4
        # Cluster path (groups_non_aligned) also defaults fp32 accumulation:
        # per-(n,g) stats reload-accumulate Ht times (512 for SDXL HW=16384) —
        # bf16 running sums quantize to rms 0.15 on (1,1,16384,320) G=32;
        # fp32 dest + Float32 stat CBs restore the budget.
        fp32_dest_acc_en = input_tensor.dtype == ttnn.float32 or groups_non_aligned
        math_approx_mode = False
        dst_full_sync_en = False

    # REDUCE_SCALAR applies the scaler twice (row then col), so 1/sqrt(N_grp)
    # turns the SUM into a mean over the HW x Cg group slab.
    n_grp = HW * Cg
    inv_sqrt_n_bits = _f32_bits(1.0 / math.sqrt(float(n_grp)))
    eps_bits = _f32_bits(eps)

    in_dtype = input_tensor.dtype
    in_page = input_tensor.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    bf16_page = ttnn.tile_size(ttnn.bfloat16)

    # Intermediate statistics format (see module docstring).
    if fp32_dest_acc_en or in_dtype == ttnn.float32:
        stat_dtype = ttnn.float32
    else:
        stat_dtype = ttnn.bfloat16
    stat_page = ttnn.tile_size(stat_dtype)

    # --- multi-core split: one work unit = one (n, g) group, or one
    # (n, cluster) on the non-aligned-groups path ---
    grid = input_tensor.device().compute_with_storage_grid_size()
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    num_units = N * num_clusters if groups_non_aligned else N * G
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(full_grid, num_units)

    CB_INPUT_TILES = 0
    CB_GAMMA_TILES = 1
    CB_BETA_TILES = 2
    CB_SCALER = 8
    CB_MASK_INTERIOR = 9
    CB_MASK_TAIL = 10
    CB_MEAN_EXPORT = 11
    CB_RSTD_EXPORT = 12
    CB_MEAN_ROW = 13
    CB_RSTD_ROW = 14
    CB_MASK_ONES = 15
    CB_MASK_ROWS = 17
    CB_OUTPUT_TILES = 16
    CB_MEAN = 24
    CB_VAR = 25
    CB_CENTERED = 26
    CB_XHAT = 27
    CB_SCALED = 28

    def cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    # Streaming row-chunk width: Wg tiles on the aligned path; on the cluster
    # path input/output/xhat/scaled are consumed tile-by-tile, so they only
    # need double-buffering at the widest per-row chunk (Ws_max or Wc_max).
    # Only cb_centered must hold a FULL chunk_rows x Ws_max block live between
    # the sub/mask/square stage and the accumulating reduce (sequential
    # helpers can't pipeline), plus pass-3 streaming headroom.
    chunk_w = max(Ws_max, Wc_max) if groups_non_aligned else Wg
    stream_pages = 2 * chunk_w
    centered_pages = max(chunk_rows * Ws_max + Ws_max, 2 * Wc_max) if groups_non_aligned else stream_pages

    # Scaler precision follows the stat format. For non-tile-aligned shapes the
    # group element count is not a power of two, so a bf16 scaler quantizes
    # 1/sqrt(N) at 2^-9 relative — the squared scaler shifts the mean by ~0.4%,
    # which the fp32 stat path can't absorb (fp32 + bf8b-gamma golden cells sit
    # at rms 0.0104 vs target 0.01). bf16 stats keep the bf16 scaler unchanged.
    scaler_dtype = stat_dtype
    scaler_page = ttnn.tile_size(scaler_dtype)

    cbs = [
        cb(CB_INPUT_TILES, stream_pages, in_page, in_dtype),
        cb(CB_SCALER, 1, scaler_page, scaler_dtype),
        cb(CB_OUTPUT_TILES, stream_pages, out_page, in_dtype),
        cb(CB_MEAN, 1, stat_page, stat_dtype),
        cb(CB_VAR, 1, stat_page, stat_dtype),
        cb(CB_CENTERED, centered_pages, stat_page, stat_dtype),
    ]
    if has_gamma:
        gamma_page = gamma_tensor.buffer_page_size()
        cbs.append(cb(CB_GAMMA_TILES, Wt, gamma_page, gamma_tensor.dtype))
        cbs.append(cb(CB_XHAT, stream_pages, stat_page, stat_dtype))
    if has_beta:
        beta_page = beta_tensor.buffer_page_size()
        cbs.append(cb(CB_BETA_TILES, Wt, beta_page, beta_tensor.dtype))
        cbs.append(cb(CB_SCALED, stream_pages, stat_page, stat_dtype))
    # pass-3 output padding zeroing (bf8b only — see compute kernel docs)
    mask_output = int(output_tensor.dtype == ttnn.bfloat8_b)
    if groups_non_aligned:
        # Per-group 0/1 column masks (interior + HW-tail variants), regenerated
        # per group. CB-wrap rule: every push frame must be the same width, so
        # frames are padded to Ws_max tiles (compute waits Wsg, pops Ws_max).
        mask_out_cluster = bool(mask_output) and hw_tail > 0
        cbs.append(cb(CB_MASK_INTERIOR, 2 * Ws_max, bf16_page, ttnn.bfloat16))
        if hw_tail > 0:
            cbs.append(cb(CB_MASK_TAIL, 2 * Ws_max, bf16_page, ttnn.bfloat16))
        # mean/rstd scalar export (compute -> reader) and reader-built per-column
        # row vectors (reader -> compute, pass 3). Row vectors always push
        # Wc_full-tile frames (zero-padded for the capped last cluster) so the
        # FIFO wraps exactly at the buffer end.
        cbs.append(cb(CB_MEAN_EXPORT, 2, stat_page, stat_dtype))
        cbs.append(cb(CB_RSTD_EXPORT, 2, stat_page, stat_dtype))
        cbs.append(cb(CB_MEAN_ROW, Wc_full, stat_page, stat_dtype))
        cbs.append(cb(CB_RSTD_ROW, Wc_full, stat_page, stat_dtype))
        if mask_out_cluster:
            # bf8b + HW tail: the output row mask is column-independent — one
            # scalar tile per variant (all-ones interior; rows < hw_tail tail).
            cbs.append(cb(CB_MASK_ONES, 1, bf16_page, ttnn.bfloat16))
            cbs.append(cb(CB_MASK_ROWS, 1, bf16_page, ttnn.bfloat16))
    else:
        # Mask rows for non-tile-aligned shapes (Refinement 2): generated once by
        # the reader, held for the whole kernel. Interior row mask (zeros in the
        # C tail columns of the last tile) and tail row mask (zeros below the HW
        # tail rows, corner combined in the last tile). bf16 — masks are 0/1.
        if c_tail > 0:
            cbs.append(cb(CB_MASK_INTERIOR, Wg, bf16_page, ttnn.bfloat16))
        if hw_tail > 0:
            cbs.append(cb(CB_MASK_TAIL, Wg, bf16_page, ttnn.bfloat16))

    # --- per-core runtime args: [start_group, num_groups_here] + addresses ---
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    start_group = 0
    for group, units_per_core in (
        (core_group_1, units_per_core_g1),
        (core_group_2, units_per_core_g2),
    ):
        for core_range in group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        gamma_tensor.buffer_address() if has_gamma else 0,
                        beta_tensor.buffer_address() if has_beta else 0,
                        start_group,
                        units_per_core,
                    ]
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        start_group,
                        units_per_core,
                    ]
                    compute_rt_args[x][y] = [start_group, units_per_core]
                    start_group += units_per_core
    assert start_group == num_units, f"work split mismatch: {start_group} != {num_units}"

    if not groups_non_aligned:
        mask_out_cluster = False
    stat_is_f32 = int(stat_dtype == ttnn.float32)

    # --- Reader: scalars first, TensorAccessorArgs at the end (input, gamma, beta) ---
    reader_ct_args = [
        Ht,
        Wt,
        Wg,
        G,
        int(has_gamma),
        int(has_beta),
        inv_sqrt_n_bits,
        hw_tail,
        c_tail,
        int(groups_non_aligned),
        cluster_ch,
        Wc_full,
        num_clusters,
        Cg,
        C,
        stat_is_f32,
        int(mask_out_cluster),
        Ws_max,
        chunk_rows,
    ]
    READER_NUM_SCALAR_CT_ARGS = len(reader_ct_args)  # accessor args start here (kernel hard-codes 19)
    assert READER_NUM_SCALAR_CT_ARGS == 19
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args()
        if has_beta
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = [Ht, Wt, Wg, G, int(groups_non_aligned), Wc_full, num_clusters, C, cluster_ch]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    # mask_output: pass-3 zeroing of output padding. Only worth doing for bf8b
    # outputs (tail tiles share block exponents between valid and padded
    # positions). For fp32/bf16 the padding is unread garbage, and the mask mul
    # costs an extra dest rounding of every valid value — measured to push the
    # fp32 + bf8b-gamma golden cells from rms 0.0075 over the 0.01 target.
    # Cluster path: padding columns are zeroed for free via rstd_row = 0, so
    # only the HW-tail rows need masking (mask_out_cluster).
    compute_ct_args = [
        Ht,
        Wt,
        Wg,
        G,
        int(has_gamma),
        int(has_beta),
        eps_bits,
        hw_tail,
        c_tail,
        int(mask_out_cluster) if groups_non_aligned else mask_output,
        int(groups_non_aligned),
        cluster_ch,
        Wc_full,
        num_clusters,
        Cg,
        C,
        Ws_max,
        chunk_rows,
    ]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=math_approx_mode,
        dst_full_sync_en=dst_full_sync_en,
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
