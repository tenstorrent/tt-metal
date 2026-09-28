// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1 = cb_in0;
    // Fused residual: out = residual + sum. The residual CB holds this core's residual shard (same tile order as
    // the output shard) and is added onto the reduced tile in DST before packing. With a bf16 DST the sum in DST is
    // already bf16, so this is the same FPU add a separate eltwise add of the packed sum performs, and the result is
    // bit-identical to it. With fp32_dest_acc the sum is not rounded to bf16 before the add, so the result can differ.
    constexpr bool fuse_residual = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t cb_residual = get_compile_time_arg_val(3);

    uint32_t rt_args_idx = 0;
    const uint32_t has_work = get_arg_val<uint32_t>(rt_args_idx++);
    if (has_work == 0) {
        return;
    }

    const uint32_t num_blocks = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    if constexpr (!fuse_residual) {
        // Hardware startup stays with the kernel; the helper owns only the summation.
        compute_kernel_hw_startup(cb_in0, cb_in0, cb_out0);

        // Sum the gathered per-device blocks resident in cb_in0 into one output block (cb_in0 is a
        // shell over the fabric-landed data and is deliberately never popped). Replaces the hand-rolled
        // pass whose DST capacity was a hardcoded `max_dst_tiles = 8` (wrong under fp32 dest-accum) and
        // whose odd-block branch was an empty "TODO: Future support" that paired blocks off the end of
        // the CB whenever num_blocks (= ring size) was odd.
        compute_kernel_lib::sum_blocks(cb_in0, cb_out0, num_blocks, block_num_tiles);
        return;
    }

    // Fused residual: the sum must stay in DST for the residual add before packing, which
    // compute_kernel_lib::sum_blocks (it packs its own output) does not expose, so this path
    // keeps the hand-rolled pass.
    const uint32_t copy_first_block = num_blocks % 2 != 0;

    constexpr uint32_t max_dst_tiles = 8;  // TODO: Make general

    CircularBuffer cb_in(cb_in0);
    CircularBuffer cb_out(cb_out0);

    cb_in.wait_front(num_blocks * block_num_tiles);
    if constexpr (fuse_residual) {
        CircularBuffer cb_res(cb_residual);
        cb_res.wait_front(block_num_tiles);
    }
    cb_out.reserve_back(block_num_tiles);

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    add_init(cb_in0, cb_in1, true);

    uint32_t num_pack_iters = (block_num_tiles + max_dst_tiles - 1) / max_dst_tiles;
    uint32_t block_num_tiles_cnt = 0;

    for (uint32_t p = 0; p < num_pack_iters; ++p) {
        uint32_t num_tiles_to_pack = std::min(max_dst_tiles, block_num_tiles - block_num_tiles_cnt);
        if constexpr (fuse_residual) {
            if (p > 0) {
                add_init(cb_in0, cb_in1, true);  // the residual add below reprogrammed unpack/math
            }
        }
        tile_regs_acquire();
        for (uint32_t block = 0; block < num_blocks; block += 2) {
            if (copy_first_block && block == 0) {
                // TODO: Future support
            } else {
                for (uint32_t i = 0; i < num_tiles_to_pack; ++i) {
                    add_tiles(
                        cb_in0,
                        cb_in1,
                        block * block_num_tiles + p * max_dst_tiles + i,
                        (block + 1) * block_num_tiles + p * max_dst_tiles + i,
                        i);
                }
            }
        }
        if constexpr (fuse_residual) {
            // DEST_TO_SRCB: sum (DST) -> SrcB, residual (CB) -> SrcA, i.e. residual + sum, the same
            // operand order as the unfused ttnn.add(residual, all_reduce_out).
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual);
            for (uint32_t i = 0; i < num_tiles_to_pack; ++i) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual, p * max_dst_tiles + i, i);
            }
        }
        tile_regs_commit();

        // Pack output tiles
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_to_pack; ++i) {
            pack_tile(i, cb_out0, p * max_dst_tiles + i);
        }
        tile_regs_release();

        block_num_tiles_cnt += num_tiles_to_pack;
    }

    cb_out.push_back(block_num_tiles);
}
