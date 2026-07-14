# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm program descriptor.

Parameterized row-parallel streaming reduction (see kernels/*.cpp). Every block
factor and buffer depth is a knob derived from one source of truth:

  * W_BLOCK_TILES  — W-tiles streamed per reduce chunk (phase-1: 1).
  * ROW_BLOCK_TILES — tile-rows per outer pass (phase-1: 1).

CB page counts and loop trip counts are computed FROM those knobs and the input
shape — never sized to Wt / W / sequence length, so per-core L1 stays bounded
for arbitrarily wide W. Phase-1 grid = single core owning all tile-rows; going
multi-core is a runtime-arg change (start_tile_row / num_tile_rows) with no
loop-nest change.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# --- Blocking-model knobs (tunable from day 1; single source of truth) ---
W_BLOCK_TILES = 1  # W-tiles streamed per reduce chunk
ROW_BLOCK_TILES = 1  # tile-rows processed per outer pass


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.ProgramDescriptor:
    shape = list(input_tensor.shape)
    origin_H = int(shape[-2])
    origin_W = int(shape[-1])
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)

    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None

    # --- Derived geometry (all from the knobs + shape) ---
    Wt = math.ceil(origin_W / TILE_DIM)
    tiles_per_image = math.ceil(origin_H / TILE_DIM)
    total_tile_rows = leading * tiles_per_image
    # Phase-1: W_BLOCK_TILES == 1 divides every Wt. (A knob > 1 that does not
    # divide Wt would need partial-last-W-block handling — a later refinement.)
    assert Wt % W_BLOCK_TILES == 0, "W_BLOCK_TILES must divide Wt (phase-1 uses 1)"
    num_w_blocks = Wt // W_BLOCK_TILES

    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0
    scaler_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    input_elt = input_tensor.element_size()
    output_elt = output_tensor.element_size()
    gamma_elt = gamma.element_size() if has_gamma else input_elt

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    gamma_dtype = gamma.dtype if has_gamma else in_dtype

    in_tile = ttnn.tile_size(in_dtype)
    out_tile = ttnn.tile_size(out_dtype)
    scaler_tile = ttnn.tile_size(ttnn.bfloat16)

    wblock_cols = W_BLOCK_TILES * TILE_DIM
    in_rm_page = wblock_cols * input_elt  # one W-block-wide stick slice
    out_rm_page = out_tile  # untilize emits tile-sized pages
    gamma_rm_page = wblock_cols * gamma_elt

    # --- Grid: phase-1 single core owns all tile-rows ---
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    start_tile_row = 0
    num_tile_rows = total_tile_rows

    # --- CB indices (semantic) ---
    CB_INPUT_RM = 0
    CB_INPUT_TILES = 1
    CB_SCALER = 2
    CB_GAMMA_RM = 3
    CB_GAMMA_TILES = 4
    CB_OUTPUT_TILES = 16
    CB_OUTPUT_RM = 17
    CB_XSQ = 24
    CB_SUMSQ = 25
    CB_NORM = 26

    # --- Buffer-depth knobs (data-movement<->compute overlap, not reuse) ---
    STICK_BLOCK = TILE_DIM  # one tile-row height of sticks
    double = 2

    def cb(index, dtype, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        # streamed input tiles (reader in TILE regime / tilize in RM regime)
        cb(CB_INPUT_TILES, in_dtype, in_tile, double * W_BLOCK_TILES),
        # reduce scaler (bf16): full [+ partial]
        cb(CB_SCALER, ttnn.bfloat16, scaler_tile, 2 if has_partial_w else 1),
        # normalized output tiles (mul -> writer in TILE / -> untilize in RM)
        cb(CB_OUTPUT_TILES, out_dtype, out_tile, double * W_BLOCK_TILES),
        # x^2 scratch (pass 1) — compute->compute, single-depth full block
        cb(CB_XSQ, in_dtype, in_tile, W_BLOCK_TILES),
        # Sum(x^2)/W accumulator -> 1/rms (held across pass 2)
        cb(CB_SUMSQ, in_dtype, in_tile, max(double * ROW_BLOCK_TILES, 2)),
    ]
    if is_row_major:
        # RM input sticks -> tilize; RM output sticks <- untilize
        cbs.append(cb(CB_INPUT_RM, in_dtype, in_rm_page, double * STICK_BLOCK))
        cbs.append(cb(CB_OUTPUT_RM, out_dtype, out_rm_page, double * W_BLOCK_TILES))
    if has_gamma:
        cbs.append(cb(CB_GAMMA_RM, gamma_dtype, gamma_rm_page, double))
        cbs.append(cb(CB_GAMMA_TILES, in_dtype, in_tile, double * W_BLOCK_TILES))
        # normalize scratch (x*(1/rms)) before gamma mul — compute->compute
        cbs.append(cb(CB_NORM, in_dtype, in_tile, W_BLOCK_TILES))

    # ================= Reader =================
    reader_ct_args = [
        1 if is_row_major else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        scaler_bits,
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        input_elt,
        gamma_elt,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address() if has_gamma else 0,
        start_tile_row,
        num_tile_rows,
    ]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ================= Writer =================
    writer_ct_args = [
        1 if is_row_major else 0,
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        output_elt,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        start_tile_row,
        num_tile_rows,
    ]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ================= Compute =================
    compute_ct_args = [
        1 if is_row_major else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
    ]
    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = [num_tile_rows, start_tile_row, eps_bits]
    compute_config = compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor()
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
