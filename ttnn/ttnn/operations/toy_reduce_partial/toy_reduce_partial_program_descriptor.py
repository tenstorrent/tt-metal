# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core streamed SUM/MAX reduction using the host planner."""

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
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    sequence = planner.make_reduce_sequence_plan(
        reductions=[
            (
                cb_in,
                planner.ReduceCallConfig(
                    input_spec=input_tensor.spec,
                    output_spec=output_tensor.spec,
                    reduce_math={"max": planner.ReduceMath.MAX, "sum": planner.ReduceMath.SUM}[pool_type],
                    reduce_dim=planner.ReduceDimension.ROW if reduce_row else planner.ReduceDimension.COLUMN,
                    scalar=1.0,
                    fp32_mode=planner.ReduceFp32Mode.FAST,
                    max_input_cb_bytes=2 * input_tensor.buffer_page_size(),
                ),
            )
        ],
        cb_ids=planner.ReduceSequenceCbIds(auxiliary_cb_id=cb_aux, accumulator_cb_id=1, output_cb_id=cb_out),
        hardware=planner.ReduceHardwareConfig(
            arch=input_tensor.device().arch(),
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
        ),
    )
    plan = sequence.calls[0].plan
    aux_dtype = ttnn.float32 if input_tensor.dtype == ttnn.float32 else ttnn.bfloat16
    role_buffers = {
        planner.ReduceCbRole.INPUT: (cb_in, input_tensor.dtype),
        planner.ReduceCbRole.AUXILIARY: (cb_aux, aux_dtype),
        planner.ReduceCbRole.OUTPUT: (cb_out, output_tensor.dtype),
    }
    cbs = []
    for requirement in plan.cb_requirements:
        cb_id, dtype = role_buffers[requirement.role]
        cbs.append(
            ttnn.CBDescriptor(
                total_size=requirement.total_size_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=requirement.page_size)
                ],
            )
        )

    reader_ct_args = [plan.Ht, plan.Wt, plan.batches, plan.chunk.output_tiles, int(reduce_row)]
    sequence.append_auxiliary_to(reader_ct_args)
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
