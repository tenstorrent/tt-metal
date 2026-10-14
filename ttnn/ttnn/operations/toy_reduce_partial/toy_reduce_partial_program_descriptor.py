# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core streamed SUM/MAX reduction using the host planner."""

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    reduce_row: bool,
    pool_type: str = "max",
) -> ttnn.ProgramDescriptor:
    planner = ttnn.reduce_planner
    cb_in, cb_aux, cb_out = 0, 2, 16
    input_cb_tiles, output_cb_tiles = 2, 2
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    sequence = planner.make_reduce_sequence_plan(
        reductions=[
            (
                cb_in,
                planner.ReduceCallConfig(
                    block=planner.ReduceBlockSpec(
                        input_tensor.shape[-2],
                        input_tensor.shape[-1],
                        input_tensor.dtype,
                        output_tensor.dtype,
                        input_cb_tiles=input_cb_tiles,
                        batches=math.prod(list(input_tensor.shape)[:-2]),
                        padded_h=input_tensor.padded_shape[-2],
                        padded_w=input_tensor.padded_shape[-1],
                        input_tile=input_tensor.spec.tile,
                        output_tile=output_tensor.spec.tile,
                    ),
                    reduce_math={"max": planner.ReduceMath.MAX, "sum": planner.ReduceMath.SUM}[pool_type],
                    reduce_dim=planner.ReduceDimension.ROW if reduce_row else planner.ReduceDimension.COLUMN,
                    scalar=1.0,
                    fp32_mode=planner.ReduceFp32Mode.FAST,
                    input_policy=planner.ReduceInputPolicy.WAIT_AND_POP_PER_TILE,
                ),
            )
        ],
        cb_ids=planner.ReduceSequenceCbIds(auxiliary_cb_id=cb_aux, accumulator_cb_id=1, output_cb_id=cb_out),
        hardware=planner.ReduceHardwareConfig(
            arch=input_tensor.device().arch(),
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )
    plan = sequence.calls[0].plan

    def cb(cb_id, dtype, page_size, pages):
        return ttnn.CBDescriptor(
            total_size=page_size * pages,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
        )

    input_tile_size = input_tensor.spec.tile.get_tile_size(input_tensor.dtype)
    output_tile_size = output_tensor.spec.tile.get_tile_size(output_tensor.dtype)
    auxiliary = sequence.auxiliary.cb
    cbs = [
        cb(cb_in, input_tensor.dtype, input_tile_size, input_cb_tiles),
        cb(cb_aux, auxiliary.data_format, auxiliary.page_size, auxiliary.page_count),
        cb(cb_out, output_tensor.dtype, output_tile_size, output_cb_tiles),
    ]

    reader_ct_args = [plan.Ht, plan.Wt, plan.batches, plan.chunk.output_tiles, int(reduce_row)]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [input_tensor.buffer_address(), 0]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct_args = [output_tensor.buffer_num_pages()]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    sequence.append_auxiliary_to(writer_ct_args)
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), 0]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=sequence.compile_time_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=cbs)
