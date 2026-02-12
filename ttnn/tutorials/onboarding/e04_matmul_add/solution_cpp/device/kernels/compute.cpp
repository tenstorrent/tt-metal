// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"

using namespace ckernel;

/**
 * @brief Compute kernel for matmul + add: output = a @ b + c
 *
 * Performs tiled matrix multiplication followed by element-wise addition.
 * For each output tile, it:
 * - Accumulates the dot product across K tiles
 * - Adds the bias tile from c
 * - Packs the result to the output circular buffer
 */
void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr tt::CBIndex cb_a = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_b = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_c = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    mm_init(cb_a, cb_b, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            // Accumulate matmul: a @ b
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);
                matmul_tiles(cb_a, cb_b, 0, 0, 0);
                cb_pop_front(cb_a, 1);
                cb_pop_front(cb_b, 1);
            }

            // Add bias from c (reusing matmul result in DST)
            cb_wait_front(cb_c, 1);
            binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c);
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c, 0, 0);
            cb_pop_front(cb_c, 1);

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();

            // Re-init matmul for next iteration (needed after binary op changed config)
            mm_init_short(cb_a, cb_b);
        }
    }
}
