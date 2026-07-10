# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core L1 benchmark for tile reduction accumulation strategies."""

import ttnn

CB_INPUT = 0
CB_OUTPUT = 16

VARIANTS = (
    "sfpu_serial_bf16",
    "fpu_dest_reuse_bf16",
    "fpu_dest_reuse_fp32",
)


_SFPU_SERIAL_KERNEL = r"""
#include <stdint.h>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t input_tiles = num_blocks * num_tiles;

    CircularBuffer input(cb_input);
    CircularBuffer output(cb_output);
    input.reserve_back(input_tiles);
    input.push_back(input_tiles);
    binary_op_init_common(cb_input, cb_input, cb_output);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        input.wait_front(input_tiles);
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            output.reserve_back(1);
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, tile, 0);
            for (uint32_t block = 1; block < num_blocks; ++block) {
                copy_tile_to_dst_init_short(cb_input);
                copy_tile(cb_input, block * num_tiles + tile, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
            output.push_back(1);
        }
        if (iter + 1 < kernel_iters) {
            output.wait_front(num_tiles);
            output.pop_front(num_tiles);
        }
    }
    input.pop_front(input_tiles);
}
"""


_FPU_DEST_REUSE_KERNEL = r"""
#include <stdint.h>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t max_dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t input_tiles = num_blocks * num_tiles;

    CircularBuffer input(cb_input);
    CircularBuffer output(cb_output);
    input.reserve_back(input_tiles);
    input.push_back(input_tiles);
    binary_op_init_common(cb_input, cb_input, cb_output);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        input.wait_front(input_tiles);
        for (uint32_t tile_base = 0; tile_base < num_tiles; tile_base += max_dst_tiles) {
            const uint32_t remaining = num_tiles - tile_base;
            const uint32_t dst_tiles = remaining < max_dst_tiles ? remaining : max_dst_tiles;
            output.reserve_back(dst_tiles);
            tile_regs_acquire();

            uint32_t first_pair = 0;
            if (num_blocks & 1) {
                copy_tile_to_dst_init_short(cb_input);
                for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                    copy_tile(cb_input, tile_base + tile, tile);
                }
                first_pair = 1;
            }

            add_tiles_init(cb_input, cb_input, true);
            for (uint32_t block = first_pair; block < num_blocks; block += 2) {
                for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                    const uint32_t lhs = block * num_tiles + tile_base + tile;
                    const uint32_t rhs = (block + 1) * num_tiles + tile_base + tile;
                    add_tiles(cb_input, cb_input, lhs, rhs, tile);
                }
            }

            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                pack_tile(tile, cb_output);
            }
            tile_regs_release();
            output.push_back(dst_tiles);
        }
        if (iter + 1 < kernel_iters) {
            output.wait_front(num_tiles);
            output.pop_front(num_tiles);
        }
    }
    input.pop_front(input_tiles);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(num_tiles):
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def create_program_descriptor(input_tensor, output_tensor, *, variant, num_blocks, num_tiles, kernel_iters=1):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if num_blocks < 2:
        raise ValueError(f"num_blocks must be at least 2, got {num_blocks}")
    if num_tiles < 1 or kernel_iters < 1:
        raise ValueError("num_tiles and kernel_iters must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.bfloat16:
        raise ValueError("tensix_all_reduce_compute supports bfloat16 tensors")
    if input_tensor.layout != ttnn.TILE_LAYOUT or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("tensix_all_reduce_compute requires TILE_LAYOUT tensors")

    expected_input = [ttnn.TILE_SIZE, num_blocks * num_tiles * ttnn.TILE_SIZE]
    expected_output = [ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE]
    if list(input_tensor.shape) != expected_input or list(output_tensor.shape) != expected_output:
        raise ValueError(f"input shape must be {expected_input} and output shape must be {expected_output}")

    fp32_dest = variant == "fpu_dest_reuse_fp32"
    source = _SFPU_SERIAL_KERNEL if variant == "sfpu_serial_bf16" else _FPU_DEST_REUSE_KERNEL
    compile_time_args = [CB_INPUT, CB_OUTPUT, num_blocks, num_tiles, kernel_iters]

    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def reduce_blocks(input_tensor, *, variant="fpu_dest_reuse_fp32", num_blocks=8, num_tiles=6, kernel_iters=1):
    """Sum corresponding tiles from contiguous blocks on one Tensix core."""
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE]),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        create_sharded_memory_config(num_tiles),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        variant=variant,
        num_blocks=num_blocks,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
