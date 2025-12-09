// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
#include "circular_buffer.h"

// Push 1 stick or partial stick to a cb (a (partial) stick consists of num_pages pages, in our case, size of a page is
// the width of a tile (setup in program factory))
inline void llk_push_pages_bilinear(const std::int32_t operand, const std::int32_t num_pages) {
    std::uint32_t output = operand;
    std::uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(output).fifo_wr_ptr += num_words;
    get_local_cb_interface(output).fifo_wr_tile_ptr = 0;

    if (get_local_cb_interface(output).fifo_wr_ptr >= get_local_cb_interface(output).fifo_limit) {
        get_local_cb_interface(output).fifo_wr_ptr -= get_local_cb_interface(output).fifo_size;
    }
}

template <uint32_t tiles_per_reduction, uint32_t unpA_face_r_dim>
inline void reduce_h_fused(const uint32_t in_cb_id, const uint32_t in_scalar_cb_id, const uint32_t out_cb_id) {
    // cb_reserve_back(out_cb_id, tiles_per_reduction);
    tile_regs_acquire();
    cb_wait_front(in_cb_id, 4);

    // Template parameters for unpack_tilizeA_B_block:
    constexpr bool use_neginf_srcA = false;  // Don't use negative infinity for source A
    constexpr bool reload_srcB = true;       // Reload source B (bilinear weights) for each operation
    constexpr bool zero_srcA = false;        // Don't zero source A
    constexpr bool zero_srcA_reduce = true;  // Zero source A for reduce operation

    // Function parameters:
    constexpr uint32_t scalar_tile_idx = 0;  // Tile index for scalar CB (only 1 tile of weights loaded)
    constexpr uint32_t num_faces = 2;  // Unpack 2 faces (top faces contain 4 rows needed for bilinear interpolation)

    unpack_tilizeA_B_block<use_neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        in_cb_id, in_scalar_cb_id, tiles_per_reduction, scalar_tile_idx, num_faces, unpA_face_r_dim);
    for (uint32_t c_i = 0; c_i < tiles_per_reduction; ++c_i) {
        reduce_tile_math(c_i, num_faces);  // Reduce the 2 faces (containing 4 rows for bilinear interpolation)
    }
    cb_pop_front(in_cb_id, 4);

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dest<tiles_per_reduction>(out_cb_id, 1, 0, 1, num_faces); /* pack 1 row (1x32) from 2 faces */
    tile_regs_release();

    PACK(llk_push_pages_bilinear(out_cb_id, tiles_per_reduction));
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t in_cb_id1 = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb_id2 = get_compile_time_arg_val(1);
    constexpr uint32_t in_scalar_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t in_scalar_cb_id2 = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(5);
    constexpr uint32_t in_ntiles_hwc = get_compile_time_arg_val(6);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(7);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(8);
    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(9);
    constexpr uint32_t blocks = get_compile_time_arg_val(10);

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    constexpr uint32_t num_output_tiles = out_ntiles_c;  //* nblocks;

    // Template parameters for tilizeA_B_reduce_init:
    constexpr bool use_neginf_srcA = false;  // Don't use negative infinity for source A
    constexpr bool zero_srcA_reduce = true;  // Zero source A for reduce operation

    // Function parameters:
    constexpr uint32_t num_faces = 2;   // Use 2 faces (top faces contain 4 rows for bilinear interpolation)
    constexpr uint32_t face_r_dim = 4;  // 4 rows per face (sufficient for bilinear interpolation)

    tilizeA_B_reduce_init<use_neginf_srcA, zero_srcA_reduce>(
        in_cb_id1, in_scalar_cb_id1, max_tiles_per_iter, out_cb_id, num_faces, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, 1, num_faces); /* pack 1 row (1x32) from 2 faces */
    for (uint32_t i = 0; i < nsticks_per_core_by_nblocks; i++) {
        const uint32_t cb_id = (i % 2 == 0) ? in_cb_id1 : in_cb_id2;
        const uint32_t scalar_cb_id = (i % 2 == 0) ? in_scalar_cb_id1 : in_scalar_cb_id2;

        for (uint32_t j = 0; j < blocks - 1; j++) {
            // Wait for the core to push data in cb
            reduce_h_fused<max_tiles_per_iter, window_size_hw>(cb_id, scalar_cb_id, out_cb_id);
            cb_pop_front(scalar_cb_id, 1);
        }
        reduce_h_fused<partial_iter_output_tiles, window_size_hw>(cb_id, scalar_cb_id, out_cb_id);
        cb_pop_front(scalar_cb_id, 1);
    }
}  // MAIN
}  // namespace NAMESPACE
