// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"

#include "sort_common.hpp"

#include "debug/dprint.h"

inline void print_loop(uint32_t count) {
    UNPACK(DPRINT << "U-LOOP:" << (uint32_t)count << ENDL());
    // MATH(DPRINT << "M-LOOP:" << (uint32_t)count << ENDL());
    // PACK(DPRINT << "P-LOOP:" << (uint32_t)count << ENDL());
}

inline void print_full_tile_column0(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "U=====!" << ENDL());
    // MATH(DPRINT << "M=====!" << ENDL());
    // PACK(DPRINT << "P=====!" << ENDL());
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
        UNPACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
        // MATH(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
        // PACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
    }
    UNPACK(DPRINT << ENDL() << "U+++++!" << ENDL());
    // MATH(DPRINT << ENDL() << "M+++++!" << ENDL());
    // PACK(DPRINT << ENDL() << "P+++++!" << ENDL());
}

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "U=====!" << ENDL());
    // MATH(DPRINT << "M=====!" << ENDL());
    // PACK(DPRINT << "P=====!" << ENDL());
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
                   << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        // MATH(
        //     DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
        //            << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        // PACK(
        //     DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
        //            << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
    }
    UNPACK(DPRINT << "U+++++!" << ENDL());
    // MATH(DPRINT << "M+++++!" << ENDL());
    // PACK(DPRINT << "P+++++!" << ENDL());
}

namespace NAMESPACE {

void MAIN {
    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t input_tensor_output_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t Ht = get_compile_time_arg_val(7);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(8);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(9);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(10);
    constexpr bool descending = get_compile_time_arg_val(11);
    constexpr bool stable =
        get_compile_time_arg_val(12);  // TODO: In the future change LLK to have the option or add additional step with
                                       // checking values and indexes after the sorting
                                       // Issue: https://github.com/tenstorrent/tt-metal/issues/20625

    // Constants
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        const bool ascending = !descending;
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Processing each row
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }
        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        // Determine direction for this comparison block
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir = ascending_block == ascending;

                        if (pair_id == processing_pair_id) {
                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i;
                            const uint32_t right_tile_id = j;

                            UNPACK(
                                DPRINT << "COMPUTE: Processing pair id: " << U32(pair_id) << " left_tile_id: "
                                       << U32(left_tile_id) << " right_tile_id: " << U32(right_tile_id)
                                       << " ascending: " << U32((uint32_t)dir) << ENDL());

                            cb_wait_front(input_tensor_cb_index, 2 * one_tile);
                            cb_wait_front(index_tensor_cb_index, 2 * one_tile);
// DPRINT << "COMPUTE: 1. INPUT TENSOR" << ENDL();
// print_full_tile(input_tensor_cb_index, left_tile_id);
                            cb_reserve_back(input_tensor_transposed_cb_index, 2 * one_tile);
                            cb_reserve_back(index_tensor_transposed_cb_index, 2 * one_tile);

                            acquire_dst();

                            reconfig_data_format_srca(input_tensor_cb_index);
                            transpose_wh_init_short(input_tensor_cb_index);
                            transpose_wh_tile(input_tensor_cb_index, 0, input_dest_start);
                            transpose_wh_tile(input_tensor_cb_index, 1, input_dest_end);

                            reconfig_data_format_srca(index_tensor_cb_index);
                            transpose_wh_init_short(index_tensor_cb_index);
                            transpose_wh_tile(index_tensor_cb_index, 0, index_dest_start);
                            transpose_wh_tile(index_tensor_cb_index, 1, index_dest_end);

                            // llk_topk_sort -> inplace
                            // ckernel::topk_local_sort(0, (int)dir, 5); // TODO: Turn on

                            // pack value tiles into transposed buffer
                            pack_reconfig_data_format(input_tensor_transposed_cb_index);
                            pack_tile(input_dest_start, input_tensor_transposed_cb_index);
                            pack_tile(input_dest_end, input_tensor_transposed_cb_index);

                            // pack index tiles into index transposed buffer
                            pack_reconfig_data_format(index_tensor_transposed_cb_index);
                            pack_tile(index_dest_start, index_tensor_transposed_cb_index);
                            pack_tile(index_dest_end, index_tensor_transposed_cb_index);

                            cb_pop_front(input_tensor_cb_index, 2 * one_tile);
                            cb_pop_front(index_tensor_cb_index, 2 * one_tile);

                            cb_push_back(input_tensor_transposed_cb_index, 2 * one_tile);
                            cb_push_back(index_tensor_transposed_cb_index, 2 * one_tile);

                            release_dst();

                            // ---------- Values tensor ------------
                            acquire_dst();

                            cb_wait_front(input_tensor_transposed_cb_index, 2 * one_tile);

                            // Transpose from sorting by column to right structure
                            reconfig_data_format_srca(input_tensor_transposed_cb_index);
                            transpose_wh_init_short(input_tensor_transposed_cb_index);
                            pack_reconfig_data_format(input_tensor_transposed_cb_index);
// {
//     volatile int delay = 100000;
//     while (delay--) {
//         asm volatile("nop");
//     }
// }
// UNPACK(DPRINT << "COMPUTE: 2. Transposed" << ENDL());
// print_full_tile(input_tensor_transposed_cb_index, 0);
// Add a small delay loop using inline assembly
// {
//     volatile int delay = 100000;
//     while (delay--) {
//         asm volatile("nop");
//     }
// }
                            cb_reserve_back(input_tensor_output_cb_index, one_tile);
                            transpose_wh_tile(input_tensor_transposed_cb_index, 0, 0);
                            pack_tile(0, input_tensor_output_cb_index);

// PACKER(DPRINT << "COMPUTE: 3. Output" << ENDL());
// print_full_tile(input_tensor_output_cb_index, 0);
                            cb_push_back(input_tensor_output_cb_index, one_tile);

                            cb_reserve_back(input_tensor_output_cb_index, one_tile);
                            transpose_wh_tile(input_tensor_transposed_cb_index, 1, 0);
                            pack_tile(0, input_tensor_output_cb_index);
                            cb_push_back(input_tensor_output_cb_index, one_tile);

                            cb_pop_front(input_tensor_transposed_cb_index, 2 * one_tile);

                            release_dst();

                            // ---------- Index tensor ------------
                            acquire_dst();

                            cb_wait_front(index_tensor_transposed_cb_index, 2 * one_tile);

                            reconfig_data_format_srca(index_tensor_transposed_cb_index);
                            transpose_wh_init_short(index_tensor_transposed_cb_index);
                            pack_reconfig_data_format(index_tensor_transposed_cb_index);

                            cb_reserve_back(index_tensor_output_cb_index, one_tile);
                            transpose_wh_tile(index_tensor_transposed_cb_index, 0, 0);
                            pack_tile(0, index_tensor_output_cb_index);
                            cb_push_back(index_tensor_output_cb_index, one_tile);

                            cb_reserve_back(index_tensor_output_cb_index, one_tile);
                            transpose_wh_tile(index_tensor_transposed_cb_index, 1, 0);
                            pack_tile(0, index_tensor_output_cb_index);
                            cb_push_back(index_tensor_output_cb_index, one_tile);

                            cb_pop_front(index_tensor_transposed_cb_index, 2 * one_tile);

                            release_dst();

                            DPRINT << "COMPUTE: Finished processing pair id: " << U32(pair_id) << ENDL();
                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop

        DPRINT << "COMPUTE: Finished processing row: " << U32(h) << ENDL();
    }  // h loop
}
}  // namespace NAMESPACE