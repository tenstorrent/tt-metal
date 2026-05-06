# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ProgramDescriptor for ttnn.operations.linear.linear (Phase 0).

Single-core matmul + optional row-broadcast bias.

CB layout (semantic name → buffer index):
    cb_input_tiles    (0)   bf16 tiles, Mt*Kt pages       — reader → compute (matmul in0)
    cb_weight_tiles   (1)   bf16 tiles, Kt*Nt pages       — reader → compute (matmul in1)
    cb_bias_tiles     (2)   bf16 tiles, Nt pages          — reader → compute (bias add)
                                                             [only when bias is provided]
    cb_output_tiles   (16)  bf16 tiles, double-buffered   — compute → writer
    cb_partials       (24)  bf16 tiles, Mt*Nt pages       — compute (matmul) → compute (bias)
                                                             [only when bias is provided]

Sequential helpers (matmul_block → add_bias_bcast_rows) cannot pipeline — both
own all 3 TRISCs — so cb_partials must hold a FULL block of Mt*Nt tiles.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# CB indices — semantic names enforced everywhere; numeric slots picked from
# the standard ranges (0-7 inputs, 16-23 outputs, 24-31 intermediates).
CB_INPUT_TILES = 0
CB_WEIGHT_TILES = 1
CB_BIAS_TILES = 2
CB_OUTPUT_TILES = 16
CB_PARTIALS = 24


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    bias: ttnn.Tensor = None,
) -> ttnn.ProgramDescriptor:
    has_bias = bias is not None

    # Tile-grid sizes derived from logical shapes (validation already enforced
    # divisibility by 32).
    M = int(input_tensor.shape[-2])
    K = int(input_tensor.shape[-1])
    N = int(weight_tensor.shape[-1])
    Mt = M // TILE_DIM
    Kt = K // TILE_DIM
    Nt = N // TILE_DIM

    # Tile sizes — every CB carries TILE-layout bf16 tiles, so all page sizes are
    # equal. We still query each tensor explicitly to keep the per-tensor metadata
    # contract from the template.
    input_page_size = input_tensor.buffer_page_size()
    weight_page_size = weight_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    input_num_pages = input_tensor.buffer_num_pages()  # = Mt * Kt
    weight_num_pages = weight_tensor.buffer_num_pages()  # = Kt * Nt
    output_num_pages = output_tensor.buffer_num_pages()  # = Mt * Nt

    bias_page_size = bias.buffer_page_size() if has_bias else 0
    bias_num_pages = bias.buffer_num_pages() if has_bias else 0  # = Nt when present

    # ------------------------------------------------------------------
    # Core grid — Phase 0 is single core (0, 0).
    # ------------------------------------------------------------------
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ------------------------------------------------------------------
    # Circular buffers
    # ------------------------------------------------------------------
    cbs = []

    # cb_input_tiles — full Mt*Kt block (matmul helper waits on the entire
    # K-block before consuming).
    cbs.append(
        ttnn.CBDescriptor(
            total_size=(Mt * Kt) * input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        )
    )

    # cb_weight_tiles — full Kt*Nt block.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=(Kt * Nt) * weight_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_WEIGHT_TILES,
                    data_format=weight_tensor.dtype,
                    page_size=weight_page_size,
                )
            ],
        )
    )

    # cb_output_tiles — double-buffered, streams to writer one tile per
    # subblock-iter.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=output_page_size,
                )
            ],
        )
    )

    if has_bias:
        # cb_bias_tiles — Nt tiles, fully populated by reader at start.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Nt * bias_page_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BIAS_TILES,
                        data_format=bias.dtype,
                        page_size=bias_page_size,
                    )
                ],
            )
        )

        # cb_partials — sized to a FULL Mt*Nt block: sequential helpers
        # (matmul_block → add_bias_bcast_rows) cannot pipeline.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=(Mt * Nt) * output_page_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_PARTIALS,
                        data_format=output_tensor.dtype,
                        page_size=output_page_size,
                    )
                ],
            )
        )

    # ------------------------------------------------------------------
    # Reader kernel — pushes input, weight, optional bias.
    # CT args (in order):
    #     [0] has_bias     (0 or 1)
    #     [1] input_num_pages   (= Mt * Kt)
    #     [2] weight_num_pages  (= Kt * Nt)
    #     [3] bias_num_pages    (= Nt when has_bias, else 0)
    # then TensorAccessorArgs: input, weight, bias  (bias unconditionally
    # declared with a no-arg placeholder when absent — matches the template
    # contract that all accessor args are at the END of the CT arg list).
    # RT args:
    #     [0] input  buffer address
    #     [1] weight buffer address
    #     [2] bias   buffer address (0 when absent)
    # ------------------------------------------------------------------
    has_bias_flag = 1 if has_bias else 0

    reader_ct_args = [has_bias_flag, input_num_pages, weight_num_pages, bias_num_pages]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(bias).get_compile_time_args()
        if has_bias
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        weight_tensor.buffer_address(),
        bias.buffer_address() if has_bias else 0,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "linear_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ------------------------------------------------------------------
    # Writer kernel — drains cb_output_tiles to DRAM, Mt*Nt tiles in
    # row-major tile order.
    # CT args:
    #     [0] num_tiles   (= Mt * Nt)
    # then TensorAccessorArgs: output.
    # RT args:
    #     [0] output buffer address
    # ------------------------------------------------------------------
    writer_ct_args = [output_num_pages]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address()]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "linear_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ------------------------------------------------------------------
    # Compute kernel — matmul_block + optional add_bias_bcast_rows.
    # CT args:
    #     [0] has_bias
    #     [1] Mt
    #     [2] Nt
    #     [3] Kt
    # ------------------------------------------------------------------
    compute_ct_args = [has_bias_flag, Mt, Nt, Kt]

    # Precision config:
    #   - HiFi4: matmul accumulates K bf16 multiplies; LoFi loses too many
    #     mantissa bits once K crosses one tile.
    #   - fp32_dest_acc_en=True: accumulate sums inside DEST as fp32, downcast
    #     to bf16 only at pack time. Without this, the K-accumulation rounds
    #     to bf16 every step and the acceptance test (atol=0.1, rtol=0.02)
    #     fails at K>=64 (max diff ~0.25 → 0.5 as K grows).
    #   With out_subblock_h*out_subblock_w = 1, DEST holds 1 fp32 tile per
    #   subblock — well under the fp32 DEST capacity.
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "linear_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
