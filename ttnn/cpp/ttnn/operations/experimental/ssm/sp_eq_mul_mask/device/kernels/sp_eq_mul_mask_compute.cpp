// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

// Fused compute kernel: output[i] = A[i] if A[i] == B[i] else 0.
// Two inputs (CB indices c_0, c_1), single output (c_2), tile-at-a-time.
//
// Flow per tile:
//   DST[0] = A      (copy_tile)
//   DST[1] = B      (copy_tile)
//   DST[2] = (DST[0] == DST[1]) ? 1 : 0   (eq_binary_tile, SFPU)
//   DST[0] = DST[0] * DST[2]              (mul_binary_tile, SFPU)
//   pack DST[0] -> output CB.
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_a, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_a);
        copy_tile(cb_a, 0, 0);  // DST[0] = A
        copy_tile_to_dst_init_short(cb_b);
        copy_tile(cb_b, 0, 1);  // DST[1] = B

        eq_binary_tile_init();
        eq_binary_tile(0, 1, 2);  // DST[2] = (A == B) ? 1 : 0

        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);  // DST[0] = A * mask

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
