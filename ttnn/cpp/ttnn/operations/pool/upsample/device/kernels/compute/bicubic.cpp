// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Bicubic upsample compute kernel.
// Processes 16 neighbors as 4 groups of 4 using bilinear-style tilize-reduce.
// Uses cb_reserve_back/cb_push_back for output CB (non-sharded ring buffer).

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    uint32_t num_output_pixels = get_arg_val<uint32_t>(0);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(3);
    constexpr uint32_t blocks = get_compile_time_arg_val(4);

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    constexpr uint32_t NUM_GROUPS = 4;
    constexpr uint32_t NEIGHBORS_PER_GROUP = 4;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    constexpr uint32_t num_faces = 2;
    constexpr uint32_t face_r_dim = NEIGHBORS_PER_GROUP;  // 4 rows per group (proven bilinear pattern)

    tilizeA_B_reduce_init<false, true>(input_cb_id, scalar_cb_id, max_tiles_per_iter, out_cb_id, num_faces, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, 1, num_faces);

    for (uint32_t pixel = 0; pixel < num_output_pixels; pixel++) {
        for (uint32_t blk = 0; blk < blocks; blk++) {
            const uint32_t tiles_this_iter = (blk == blocks - 1) ? partial_iter_output_tiles : max_tiles_per_iter;

            tile_regs_acquire();

            // Process 4 groups of 4 neighbors — accumulate in dest registers
            for (uint32_t g = 0; g < NUM_GROUPS; g++) {
                cb_wait_front(input_cb_id, NEIGHBORS_PER_GROUP);

                unpack_tilizeA_B_block<
                    false,  // use_neginf_srcA
                    true,   // reload_srcB
                    false,  // zero_srcA
                    true    // zero_srcA_reduce
                    >(input_cb_id, scalar_cb_id, tiles_this_iter, 0, num_faces, face_r_dim);

                for (uint32_t c_i = 0; c_i < tiles_this_iter; ++c_i) {
                    reduce_tile_math(c_i, num_faces);
                }

                cb_pop_front(input_cb_id, NEIGHBORS_PER_GROUP);
                cb_pop_front(scalar_cb_id, 1);
            }

            tile_regs_wait();
            tile_regs_commit();

            // Reserve output CB space BEFORE pack (required for non-sharded ring-buffer CBs)
            cb_reserve_back(out_cb_id, tiles_this_iter);

            if constexpr (max_tiles_per_iter == partial_iter_output_tiles) {
                pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, 1, num_faces);
            } else {
                if (tiles_this_iter == max_tiles_per_iter) {
                    pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, 1, num_faces);
                } else {
                    pack_untilize_dest<partial_iter_output_tiles>(out_cb_id, 1, 0, 1, num_faces);
                }
            }

            tile_regs_release();

            // Push output pages using standard CB API (not llk_push_pages)
            cb_push_back(out_cb_id, tiles_this_iter);
        }
    }
}
