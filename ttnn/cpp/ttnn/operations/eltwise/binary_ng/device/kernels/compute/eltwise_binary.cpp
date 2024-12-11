// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include <cstdint>


namespace NAMESPACE {

ALWI void process_tile(uint32_t cb_bcast, uint32_t cb_other, uint32_t cb_out, uint32_t freq, uint32_t tile_start) {
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_bcast, onetile);

    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(cb_other, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();
        add_tiles(cb_bcast, cb_other, 0, 0, 0);
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_other, onetile);
    }
    cb_pop_front(cb_bcast, onetile);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

#if BCAST_INPUT
    auto cb_bcast = cb_in1;
    auto cb_other = cb_in0;
#else
    auto cb_bcast = cb_in0;
    auto cb_other = cb_in1;
#endif

    binary_op_init_common(cb_bcast, cb_other, cb_out0);
    add_tiles_init();

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_bcast, cb_other, cb_out0, tile_freq, tile_start);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_bcast, cb_other, cb_out0, remaining_iterations, tile_start);
    }
}
}  // namespace NAMESPACE
