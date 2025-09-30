// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

void print_tile(uint32_t cb_idx, uint32_t tile_idx, bool untilize = false) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

namespace NAMESPACE {

void MAIN {
    // Grid sample specific compile-time arguments
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);  // REDUCTION_SIZE for grid sample
    constexpr uint32_t split_reader = get_compile_time_arg_val(2);
    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);  // for split reader
    // Skip unused indices CBs (9, 10)
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(11);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(12);
    // Skip unused temp CBs (13, 14)
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(15);
    // Skip unused out_idx_cb_id (16)
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(17);
    // Skip return_indices (18) - always false for grid sample
    constexpr uint32_t pre_tilize_cb_id = get_compile_time_arg_val(19);  // unused for grid sample
    // Skip is_output_tiled (20) - always false for grid sample
    // Skip is_output_block_format (21) - always false for grid sample

    // Grid sample constants
    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT ? window_size_hw : FACE_HEIGHT;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_HEIGHT || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = 2;
    constexpr uint32_t num_faces_in_last_output_tile = last_tile_is_partial && in_c % TILE_WIDTH <= FACE_WIDTH ? 1 : 2;
    constexpr uint32_t num_out_sticks = 1;

    // Grid sample uses SUM reduction (for bilinear interpolation averaging)
    constexpr bool neginf_srca_maxpool = false;  // No max pool for grid sample
    constexpr bool zero_srca_avgpool = true;     // Use zero init for sum reduction

    // Simplified tile processing for grid sample - no large kernel complexity needed
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;  // Grid sample doesn't need the complex large kernel logic
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    // Simplified tilize reconfiguration - only when beneficial and safe
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     window_size_hw <= FACE_HEIGHT && !last_tile_is_partial;

    // Initialize for grid sample (SUM reduction, untilized output)
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, out_cb_id, num_faces_in_input_tile, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_sticks, num_faces_in_output_tile);

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // Wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }

    // Main processing loop for grid sample interpolations
    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        const bool reader0 = !(split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;

        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
        }

        // Process channel blocks
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            const uint32_t number_of_tiles = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            const uint32_t output_faces =
                (last_tile_is_partial && last_c_block)
                    ? (number_of_tiles - 1) * num_faces_in_output_tile + num_faces_in_last_output_tile
                    : number_of_tiles * num_faces_in_output_tile;

            // Reserve output space
            cb_reserve_back(out_cb_id, output_faces);

            // Reconfigure tilize if needed for this block
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce, num_faces_in_input_tile, face_r_dim, 1)));
                }
            }

            tile_regs_acquire();

            // Process reduction chunks for bilinear interpolation
            // for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
            cb_wait_front(curr_in_cb_id, 1);

            // Perform tilized reduction (sum for bilinear interpolation)
            unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                curr_in_cb_id,
                curr_scalar_cb_id,
                tiles_to_reduce,
                0,  // tile idx for Src b is 0 because only 1 tile of constants is loaded
                num_faces_in_input_tile,
                face_r_dim);

            for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                reduce_tile_math(math_tile_idx, num_faces_in_input_tile);
            }

            if (n == 0) {
                tensix_sync();
                dprint_tensix_dest_reg(0);
                tensix_sync();
                // dprint_tensix_dest_reg(1);
            }

            cb_pop_front(curr_in_cb_id, 1);

            tile_regs_commit();
            tile_regs_wait();

            // Pack output directly to row-major format (no tiling needed for grid sample)
            if (n == 0) {
                tensix_sync();
                dprint_tensix_dest_reg(0);
                tensix_sync();
            }

            if (last_c_block) {
                pack_untilize_dest<partial_iter_output_tiles>(
                    out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
            } else {
                pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
            }

            cb_push_back(out_cb_id, output_faces);
            tile_regs_release();
        }

        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
}

}  // namespace NAMESPACE
