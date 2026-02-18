// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    mm_init(cb_in0, cb_in1, cb_out);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                acquire_dst();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    cb_wait_front(cb_in0, onetile);
                    cb_wait_front(cb_in1, onetile);

                    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                    cb_pop_front(cb_in0, onetile);
                    cb_pop_front(cb_in1, onetile);
                }

                cb_reserve_back(cb_out, onetile);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, onetile);

                release_dst();
            }
        }
    }
}
