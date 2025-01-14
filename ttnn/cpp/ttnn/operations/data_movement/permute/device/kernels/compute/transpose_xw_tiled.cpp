// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

template <uint32_t N>
void dprint_array(const uint32_t* arr, const char* name) {
    DPRINT << name << ": ";
    for (uint32_t i = 0; i < N; i++) {
        DPRINT << arr[i] << " ";
    }
    DPRINT << ENDL();
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    uint32_t start_block = get_arg_val<uint32_t>(0);
    uint32_t end_block = get_arg_val<uint32_t>(1);
    uint32_t input_shape[N], dims[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + 2);
        dims[i] = get_arg_val<uint32_t>(i + N + 2);
    }

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;

    uint32_t x_dim = dims[N - 1];
    uint32_t x = input_shape[x_dim];
    uint32_t w = input_shape[N - 1];

    uint32_t X_p = TILE_HEIGHT * ((x + TILE_HEIGHT - 1) / TILE_HEIGHT);
    uint32_t W_p = TILE_WIDTH * ((w + TILE_WIDTH - 1) / TILE_WIDTH);

    uint32_t padded_xw_volume = X_p * W_p;
    for (uint32_t i = 0; i < N - 1; i++) {
        if (i == x_dim) {
            continue;
        }
        padded_xw_volume *= input_shape[i];
    }

    uint32_t xw_blocks = padded_xw_volume / (TILE_HEIGHT * TILE_WIDTH);
    end_block = xw_blocks;

    constexpr uint32_t x_block_size = TILE_HEIGHT;
    constexpr uint32_t w_block_size = TILE_WIDTH;

    uint32_t w_blocks = W_p / w_block_size;
    uint32_t x_blocks = X_p / x_block_size;

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // unary_op_init_common(cb_in, cb_out);
    UNPACK(DPRINT << "N: " << N << ENDL());
    UNPACK(DPRINT << "start_block: " << start_block << ENDL());
    UNPACK(DPRINT << "end_block: " << end_block << ENDL());
    UNPACK(dprint_array<N>(input_shape, "input_shape"));
    UNPACK(dprint_array<N>(dims, "dims"));

    for (uint32_t block = start_block; block < end_block; block++) {
        // Decompose block into w_block, x_block, and xw_block indices
        uint32_t rem = block;
        uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t h = rem % input_shape[N - 2];
        UNPACK(DPRINT << "h: " << h << ENDL());
        // tilize input via unpack and then pack
        // tilize_init_short(cb_in, 1, cb_tilize);

        // cb_wait_front(cb_in, 1);
        // cb_reserve_back(cb_tilize, 1);

        // tilize_block(cb_in, 1, cb_tilize);  // tilize and pack into cb_tilize

        // cb_push_back(cb_tilize, 1);
        // cb_pop_front(cb_in, 1);

        // tilize_uninit(cb_in, cb_tilize);

        // // transpose input
        // cb_wait_front(cb_tilize, 1);
        // transpose_wh_init_short(cb_tilize);
        // pack_untilize_dst_init_short<1>(cb_out);

        // tile_regs_acquire();
        // transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        // tile_regs_commit();

        // // pack and untilize
        // cb_reserve_back(cb_out, 1);

        // tile_regs_wait();
        // pack_untilize_dst<1>(cb_out);  // pack call
        // tile_regs_release();

        // cb_push_back(cb_out, 1);

        // cb_wait_front(cb_out, 1);
        // pack_untilize_uninit(cb_out);

        // cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
