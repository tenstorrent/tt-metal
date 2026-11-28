# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for g6_unsqueeze: one mover kernel per data-movement RISC, no compute.

Every extent comes from the plan (`g6_unsqueeze_plan` in the op file); this module only
turns a plan into CBs, kernels and arguments. Buffer addresses are the only per-call data
and ride in each kernel's common runtime args, so a cached descriptor is reused by
patching those two words.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
MOVER_KERNEL = KERNEL_DIR / "g6_unsqueeze_mover.cpp"

# Common runtime arg slots of every mover.
COMMON_ARG_INPUT_ADDR = 0
COMMON_ARG_OUTPUT_ADDR = 1


def _core_range_set(ranges):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in ranges]
    )


def build_program_descriptor(plan, input_tensor, output_tensor):
    geometry = plan.geometry
    mover_cores = _core_range_set(plan.core_ranges)
    face_slots_pages = plan.staging_depth * plan.block_w_tiles * plan.block_tile_halves

    per_core_args = ttnn.RuntimeArgs()
    for (x, y), first_tile, core_w_tiles in plan.core_work:
        per_core_args[x][y] = [first_tile, core_w_tiles]

    cbs = []
    kernels = []
    for role in plan.roles:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=face_slots_pages * geometry.face_bytes,
                core_ranges=mover_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=role.cb_index,
                        data_format=input_tensor.dtype,
                        page_size=geometry.face_bytes,
                    )
                ],
            )
        )
        compile_time_args = [
            role.cb_index,
            role.first_half,
            plan.block_tile_halves,
            plan.block_w_tiles,
            plan.staging_depth,
            geometry.face_h,
            geometry.face_w,
            geometry.face_bytes,
            geometry.face_row_bytes,
            geometry.faces_per_face_row * geometry.face_bytes,
            input_tensor.buffer_aligned_page_size(),
            output_tensor.buffer_aligned_page_size(),
        ]
        compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

        common_args = [0, 0]
        common_args[COMMON_ARG_INPUT_ADDR] = input_tensor.buffer_address()
        common_args[COMMON_ARG_OUTPUT_ADDR] = output_tensor.buffer_address()

        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(MOVER_KERNEL),
                core_ranges=mover_cores,
                compile_time_args=compile_time_args,
                runtime_args=per_core_args,
                common_runtime_args=common_args,
                config=ttnn.DataMovementConfigDescriptor(processor=role.processor, noc=role.noc),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def patch_buffer_addresses(program_descriptor, input_tensor, output_tensor):
    """Point a cached descriptor at this call's buffers; nothing else changes on a cache hit."""
    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    # `kernels` yields references into the descriptor, so this writes the cached object in place.
    for kernel in program_descriptor.kernels:
        kernel.common_runtime_args[COMMON_ARG_INPUT_ADDR] = input_addr
        kernel.common_runtime_args[COMMON_ARG_OUTPUT_ADDR] = output_addr
