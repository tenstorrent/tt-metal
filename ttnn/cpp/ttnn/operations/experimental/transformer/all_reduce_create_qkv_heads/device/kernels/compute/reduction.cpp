// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1 = cb_in0;

    uint32_t rt_args_idx = 0;
    const uint32_t num_blocks = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t copy_first_block = num_blocks % 2 != 0;
    constexpr uint32_t max_dst_tiles = 8;  // TODO: Make general

    cb_wait_front(cb_in0, num_blocks * block_num_tiles);
    cb_reserve_back(cb_out0, block_num_tiles);

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1, true);

    uint32_t num_pack_iters = (block_num_tiles + max_dst_tiles - 1) / max_dst_tiles;
    uint32_t block_num_tiles_cnt = 0;

    for (uint32_t p = 0; p < num_pack_iters; ++p) {
        uint32_t num_tiles_to_pack = std::min(max_dst_tiles, block_num_tiles - block_num_tiles_cnt);
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
        tile_regs_commit();

        // Pack output tiles
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_to_pack; ++i) {
            pack_tile(i, cb_out0, p * max_dst_tiles + i);
        }
        tile_regs_release();

        block_num_tiles_cnt += num_tiles_to_pack;
    }

    cb_push_back(cb_out0, block_num_tiles);
}
}  // namespace NAMESPACE
