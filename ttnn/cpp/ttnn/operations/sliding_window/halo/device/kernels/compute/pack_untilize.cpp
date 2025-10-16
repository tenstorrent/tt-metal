// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id0 = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(src_cb_id, out_cb_id0);

#ifdef RUNTIME_TILES_PER_ROW
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(1);  // number of tiles along width of shard

    constexpr bool use_pack_untilize = false;

    untilize_init(src_cb_id);
    const uint32_t tiles_per_block = block_size * tiles_per_row;
#else
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);  // number of tiles along width of shard
    constexpr bool use_pack_untilize = tiles_per_row <= MAX_PACK_UNTILIZE_WIDTH;

    if constexpr (use_pack_untilize) {
        pack_untilize_init<tiles_per_row>(src_cb_id, out_cb_id0);
    } else {
        untilize_init(src_cb_id);
    }
    constexpr uint32_t tiles_per_block = block_size * tiles_per_row;
#endif

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        const uint32_t out_cb_id = (block_idx % NUM_RISCV_DATA_MOVEMENT_CORES == 0) ? out_cb_id0 : out_cb_id1;

        cb_wait_front(src_cb_id, tiles_per_block);
        cb_reserve_back(out_cb_id, tiles_per_block);
#if RUNTIME_TILES_PER_ROW
        untilize_block(src_cb_id, block_size * tiles_per_row, out_cb_id);
#else
        if constexpr (use_pack_untilize) {
            pack_untilize_block<tiles_per_row>(src_cb_id, block_size, out_cb_id);
        } else {
            untilize_block(src_cb_id, block_size * tiles_per_row, out_cb_id);
        }
#endif
        cb_push_back(out_cb_id, tiles_per_block);
        cb_pop_front(src_cb_id, tiles_per_block);
    }
    if constexpr (use_pack_untilize) {
        pack_untilize_uninit(out_cb_id0);
    }
}
}  // namespace NAMESPACE
