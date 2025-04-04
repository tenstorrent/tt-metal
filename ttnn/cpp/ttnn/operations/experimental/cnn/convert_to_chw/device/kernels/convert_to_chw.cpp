// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/transpose_wh.h"

template <int BATCH_SIZE>
FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, BATCH_SIZE);

    tile_regs_acquire();
    for (uint32_t i = 0; i < BATCH_SIZE; i++) {
        transpose_wh_tile(cb_in, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_in, BATCH_SIZE);

    cb_reserve_back(cb_out, BATCH_SIZE);
    tile_regs_wait();
    pack_untilize_dst<1>(cb_out, BATCH_SIZE);
    tile_regs_release();

    cb_push_back(cb_out, BATCH_SIZE);
}
namespace NAMESPACE {
void MAIN {
    constexpr int BATCH_SIZE = 8;
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_batches = total_tiles / BATCH_SIZE;
    const uint32_t leftover = total_tiles % BATCH_SIZE;
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_transpose_in = get_compile_time_arg_val(1);

    pack_untilize_init(cb_in, cb_transpose_in);
    transpose_wh_init(cb_in, cb_transpose_in);

    pack_untilize_dst_init_short<1>(cb_transpose_in);

    for (uint32_t i = 0; i < num_batches; i++) {
        transpose<BATCH_SIZE>(cb_in, cb_transpose_in);
    }

    for (uint32_t idx = 0; idx < leftover; idx++) {
        transpose<1>(cb_in, cb_transpose_in);
    }
    pack_untilize_uninit(cb_transpose_in);
}
}  // namespace NAMESPACE
