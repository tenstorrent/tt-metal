# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

Row-parallel, bounded two-pass streaming reduce over W (op_design.md §1, §5-§9):

  Pass 1: x -> square -> accumulate_reduce(SUM, 1/W)  -> cb_rstd = mean(x^2)
          cb_rstd -> (+eps, rsqrt)                    -> cb_rstd = 1/RMS  (held)
  Pass 2: x -> mul<Col>(x, rstd)                       -> cb_norm
          cb_norm -> mul<Row>(norm, gamma) / copy      -> cb_out

Work distribution: the R = NC*ceil(H/32) independent tile-rows are split across
the whole compute grid via `split_work_to_cores(R, grid, row_wise=True)`; each
core owns a contiguous run [row_start, row_start+num_rows) and loops its rows,
each row a 2-pass streaming reduce over NUM_BLOCKS blocks of BLOCK_SIZE tiles.

Every block knob is a live parameter (single source of truth):
  BLOCK_SIZE = pick_block_size(Wt)   -> CT arg (reader, compute)
  NUM_BLOCKS = Wt // BLOCK_SIZE       -> derived
  DEPTH      = 2                      -> CB depth (num_pages = DEPTH*BLOCK_SIZE)
No CB is sized by an op dimension (Wt/W/H/R); cb_rstd is 1 tile per row.

RM regime uses the tilize/untilize dataflow helpers
(dataflow_kernel_lib::read_sticks_for_tilize / write_sticks_after_untilize),
which handle non-tile-aligned W (row_bytes) and H (partial last block) and the
per-core start-row offset natively — no host-side pad/slice.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32
DEPTH = 2  # per-streaming-CB double-buffer depth (op_design.md §1)


def _pick_block_size(Wt: int) -> int:
    """Largest divisor of Wt that is <= 8 (the double_buffer sweet spot; not 1).

    Phase-1 value of the BLOCK_SIZE knob (op_design.md §1). Kept a function so a
    later refinement can raise the cap / change the policy in one place.
    """
    for candidate in range(min(8, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: "ttnn.Tensor | None" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    shape = list(input_tensor.shape)

    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    # Alignment-aware tile geometry (per-image ceil; op_design.md §6).
    Ht_img = ttnn.div_up(origin_H, TILE_DIM)
    Wt = ttnn.div_up(origin_W, TILE_DIM)
    R = NC * Ht_img
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    BLOCK_SIZE = _pick_block_size(Wt)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None

    inv_N_bits = _f32_bits(1.0 / float(origin_W))  # scaler = 1/origin_W (true element count)
    eps_bits = _f32_bits(epsilon)

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    in_elem = input_tensor.element_size()
    out_elem = output_tensor.element_size()
    in_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_elem = gamma.element_size()
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
    else:
        gamma_dtype = in_dtype
        gamma_elem = in_elem
        gamma_page = in_page
        tile_gamma = tile_in

    # ---- Work distribution: split R tile-rows across the whole grid ----
    grid_size = device.compute_with_storage_grid_size()
    (
        _num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, R, row_wise=True)

    assignment = []  # (core, row_start, num_rows)
    start = 0
    for group, per_core in ((core_group_1, rows_per_core_g1), (core_group_2, rows_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core

    # ---- Circular buffers (op_design.md §7); no CB sized by an op dimension ----
    CB_X_STICKS = 0
    CB_X_IN = 1
    CB_SCALER = 2
    CB_GAMMA = 3
    CB_GAMMA_STICKS = 4
    CB_OUT = 16
    CB_OUT_STICKS = 17
    CB_XSQ = 24
    CB_RSTD = 25
    CB_NORM = 26

    cbs = []

    def add_cb(idx, page_size, num_pages, fmt):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    # streaming input tiles (both passes)
    add_cb(CB_X_IN, tile_in, DEPTH * BLOCK_SIZE, in_dtype)
    # reduce scaler: 1/W (+ partial tile), bf16, wait-not-pop across all rows
    add_cb(CB_SCALER, tile_bf16, 2 if has_partial_w else 1, ttnn.bfloat16)
    # pass-1 intermediate x^2 (square -> reduce), sequential -> holds a full block
    add_cb(CB_XSQ, tile_in, 2 * BLOCK_SIZE, in_dtype)
    # 1/RMS (1 tile/row), fp32 accumulate, held across pass 2
    add_cb(CB_RSTD, tile_fp32, max(DEPTH, 2), ttnn.float32)
    # pass-2 intermediate x*rstd (mul<Col> -> mul<Row>), sequential -> full block
    add_cb(CB_NORM, tile_in, 2 * BLOCK_SIZE, in_dtype)
    # output tiles (TILE: -> writer; RM: -> untilize)
    add_cb(CB_OUT, tile_out, DEPTH * BLOCK_SIZE, out_dtype)

    if is_rm:
        # RM x: raw sticks packed for compute-side tilize (tile-paged, TILE granularity)
        add_cb(CB_X_STICKS, tile_in, DEPTH * BLOCK_SIZE, in_dtype)
        # RM out: untilized row-major (tile-paged output of compute untilize)
        add_cb(CB_OUT_STICKS, tile_out, DEPTH * BLOCK_SIZE, out_dtype)

    if has_gamma:
        add_cb(CB_GAMMA, tile_gamma, DEPTH * BLOCK_SIZE, gamma_dtype)
        add_cb(CB_GAMMA_STICKS, tile_gamma, DEPTH * BLOCK_SIZE, gamma_dtype)

    # ---- Reader kernel ----
    reader_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        1 if is_rm else 0,
        1 if has_gamma else 0,
        in_elem,
        gamma_elem,
        in_page,
        gamma_page,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    for core, row_start, num_rows in assignment:
        reader_rt_args[core.x][core.y] = [in_addr, gamma_addr, row_start, num_rows]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel ----
    writer_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        1 if is_rm else 0,
        out_elem,
        out_page,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    out_addr = output_tensor.buffer_address()
    for core, row_start, num_rows in assignment:
        writer_rt_args[core.x][core.y] = [out_addr, row_start, num_rows]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel ----
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        1 if is_rm else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        eps_bits,
    ]

    compute_rt_args = ttnn.RuntimeArgs()
    for core, row_start, num_rows in assignment:
        compute_rt_args[core.x][core.y] = [num_rows]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
