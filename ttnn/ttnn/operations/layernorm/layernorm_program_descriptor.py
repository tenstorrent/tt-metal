# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import struct
from pathlib import Path
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices
CB_IN = 0
CB_GAMMA = 1
CB_BETA = 2
CB_GAMMA_RM = 3
CB_BETA_RM = 4
CB_OUT_TILE = 16
CB_OUT_RM = 17
CB_TILIZED = 24
CB_SCALER = 25
CB_EPS = 26
CB_MEAN = 27
CB_CENTERED = 28
CB_SQ = 29
CB_VAR = 30
CB_INV_STD = 31


def _float_to_bits(f: float) -> int:
    return struct.unpack("I", struct.pack("f", f))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    eps: float,
) -> ttnn.ProgramDescriptor:
    # Tensor metadata
    W = input_tensor.shape[-1]
    Wt = W // 32
    tile_size = ttnn.tile_size(input_tensor.dtype)
    rm_page_size = W * 2  # bf16 = 2 bytes per element

    # Flatten all dims except last into rows, then count tile rows (groups of 32)
    total_elements = 1
    for d in input_tensor.shape:
        total_elements *= d
    num_rows = total_elements // W
    num_tile_rows = num_rows // 32
    num_rm_pages = num_rows  # one RM page per row

    # Core grid - single core
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ==================== Circular Buffers ====================
    cb_descriptors = []

    def make_cb(index, page_size, total_size, dtype=ttnn.bfloat16):
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=total_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index,
                        data_format=dtype,
                        page_size=page_size,
                    )
                ],
            )
        )

    # cb_in: input only (32 pages per tile row)
    make_cb(CB_IN, rm_page_size, 32 * rm_page_size)
    # cb_gamma: persistent tilized gamma
    make_cb(CB_GAMMA, tile_size, Wt * tile_size)
    # cb_beta: persistent tilized beta
    make_cb(CB_BETA, tile_size, Wt * tile_size)
    # cb_gamma_rm: RM gamma (1 page)
    make_cb(CB_GAMMA_RM, rm_page_size, rm_page_size)
    # cb_beta_rm: RM beta (1 page)
    make_cb(CB_BETA_RM, rm_page_size, rm_page_size)
    # cb_out_tile: normalized+scaled tiles
    make_cb(CB_OUT_TILE, tile_size, Wt * tile_size)
    # cb_out_rm: untilize output (must have tile-sized pages for pack_untilize)
    make_cb(CB_OUT_RM, tile_size, Wt * tile_size)
    # cb_tilized: persistent tilized input
    make_cb(CB_TILIZED, tile_size, Wt * tile_size)
    # cb_scaler: 1/W scaler, single tile
    make_cb(CB_SCALER, tile_size, tile_size)
    # cb_eps: epsilon, single tile
    make_cb(CB_EPS, tile_size, tile_size)
    # cb_mean: mean column, single tile
    make_cb(CB_MEAN, tile_size, tile_size)
    # cb_centered: persistent (x-mean)
    make_cb(CB_CENTERED, tile_size, Wt * tile_size)
    # cb_sq: (x-mean)^2, reused for gamma*norm
    make_cb(CB_SQ, tile_size, Wt * tile_size)
    # cb_var: variance column, single tile
    make_cb(CB_VAR, tile_size, tile_size)
    # cb_inv_std: rsqrt(var+eps), single tile
    make_cb(CB_INV_STD, tile_size, tile_size)

    # ==================== Reader Kernel ====================
    eps_bits = _float_to_bits(eps)
    reader_ct_args = [W, eps_bits]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    gamma_ta = ttnn.TensorAccessorArgs(gamma)
    reader_ct_args.extend(gamma_ta.get_compile_time_args())
    beta_ta = ttnn.TensorAccessorArgs(beta)
    reader_ct_args.extend(beta_ta.get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_rm_pages,
        0,  # start_page_id for input
        gamma.buffer_address(),
        1,  # num_pages for gamma (1 RM page)
        beta.buffer_address(),
        1,  # num_pages for beta (1 RM page)
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layernorm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ==================== Writer Kernel ====================
    writer_ct_args = [CB_OUT_RM, Wt, rm_page_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_rm_pages,
        0,  # start_page_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layernorm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ==================== Compute Kernel ====================
    compute_ct_args = [Wt, num_tile_rows]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layernorm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )
