// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
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
    constexpr uint32_t log2Wt = get_compile_time_arg_val(13);
    constexpr bool is_row_major = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t rm_input_value_cb_index = get_compile_time_arg_val(15);
    constexpr uint32_t rm_input_index_cb_index = get_compile_time_arg_val(16);
    constexpr uint32_t rm_output_value_cb_index = get_compile_time_arg_val(17);
    constexpr uint32_t rm_output_index_cb_index = get_compile_time_arg_val(18);

    // Constants
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    constexpr uint32_t TILE_H = 32;

    // For ROW_MAJOR, compute_kernel_hw_startup initialises the MATH-PACK DST
    // semaphore required by tilize_block before the first pair is processed.
    // For TILE layout the existing topk_tile_init + transpose_wh_init path is used.
    if constexpr (is_row_major) {
        compute_kernel_hw_startup(rm_input_value_cb_index, rm_input_index_cb_index, input_tensor_cb_index);
    } else {
        ckernel::topk_tile_init();
        transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);
    }

    for (uint32_t h = 0; h < Ht; h++) {
        const bool ascending = !descending;
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        for (uint32_t stage = 1; stage <= log2Wt; stage++) {
            const uint32_t m_iter = stage - 1;
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir = ascending_block == ascending;

                        if (pair_id == processing_pair_id) {
                            if constexpr (is_row_major) {
                                tilize_init(rm_input_value_cb_index, 2, input_tensor_cb_index);
                                cb_wait_front(rm_input_value_cb_index, 2 * TILE_H);
                                cb_reserve_back(input_tensor_cb_index, 2);
                                tilize_block(rm_input_value_cb_index, 2, input_tensor_cb_index);
                                cb_push_back(input_tensor_cb_index, 2);
                                cb_pop_front(rm_input_value_cb_index, 2 * TILE_H);
                                tilize_uninit(rm_input_value_cb_index, input_tensor_cb_index);
                                binary_op_init_common(
                                    rm_input_index_cb_index, rm_input_index_cb_index, index_tensor_cb_index);

                                tilize_init(rm_input_index_cb_index, 2, index_tensor_cb_index);
                                cb_wait_front(rm_input_index_cb_index, 2 * TILE_H);
                                cb_reserve_back(index_tensor_cb_index, 2);
                                tilize_block(rm_input_index_cb_index, 2, index_tensor_cb_index);
                                cb_push_back(index_tensor_cb_index, 2);
                                cb_pop_front(rm_input_index_cb_index, 2 * TILE_H);
                                tilize_uninit(rm_input_index_cb_index, index_tensor_cb_index);
                                binary_op_init_common(
                                    input_tensor_cb_index, index_tensor_cb_index, input_tensor_transposed_cb_index);

                                ckernel::topk_tile_init();
                                transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);
                            }

                            cb_wait_front(input_tensor_cb_index, 2 * one_tile);
                            cb_wait_front(index_tensor_cb_index, 2 * one_tile);

                            tile_regs_acquire();
                            // For RM, tiles from tilize are always regular (non-transposed),
                            // so we always transpose before sort (same as the stage==1 sub==1
                            // branch in the TILE path).  For TILE, tiles are pre-transposed
                            // in all stages except the very first.
                            if ((stage == 1 && sub == 1) || is_row_major) {
                                reconfig_data_format_srca(input_tensor_cb_index);
                                transpose_wh_init_short(input_tensor_cb_index);
                                transpose_wh_tile(input_tensor_cb_index, 0, input_dest_start);
                                transpose_wh_tile(input_tensor_cb_index, 1, input_dest_end);

                                // Process index tiles
                                reconfig_data_format_srca(index_tensor_cb_index);
                                transpose_wh_init_short(index_tensor_cb_index);
                                transpose_wh_tile(index_tensor_cb_index, 0, index_dest_start);
                                transpose_wh_tile(index_tensor_cb_index, 1, index_dest_end);
                            } else {
                                // Intermediate step - tiles are already transposed
                                // Process value tiles
                                reconfig_data_format_srca(input_tensor_cb_index);
                                copy_tile_to_dst_init_short(input_tensor_cb_index);
                                copy_tile(input_tensor_cb_index, 0, input_dest_start);
                                copy_tile(input_tensor_cb_index, 1, input_dest_end);

                                // Process index tiles
                                reconfig_data_format_srca(index_tensor_cb_index);
                                copy_tile_to_dst_init_short(index_tensor_cb_index);
                                copy_tile(index_tensor_cb_index, 0, index_dest_start);
                                copy_tile(index_tensor_cb_index, 1, index_dest_end);
                            }

                            cb_pop_front(input_tensor_cb_index, 2 * one_tile);
                            cb_pop_front(index_tensor_cb_index, 2 * one_tile);

                            uint32_t tile_input_low = input_dest_start;
                            uint32_t tile_input_high = input_dest_end;
                            uint32_t tile_index_low = index_dest_start;
                            uint32_t tile_index_high = index_dest_end;

                            if (sub == 1) {
                                // Use sort LLK only the last substage to sort the last pair of tiles - speed up
                                ckernel::topk_local_sort(/*idst=*/0, (int)dir, /*end_phase(log2(K))=*/5);
                            } else {
                                // For all other stages use topk_merge to put the top K values in one tile, and the
                                // bottom K values in another tile
                                ckernel::topk_merge(/*idst=*/0, m_iter, /*k=*/32);

                                // topk_merge puts smallest values in DEST[0] and largest in DEST[1]
                                // We swap their indices when using descending order
                                if (dir) {
                                    tile_input_low = input_dest_end;
                                    tile_input_high = input_dest_start;
                                    tile_index_low = index_dest_end;
                                    tile_index_high = index_dest_start;
                                }
                            }

                            tile_regs_commit();
                            tile_regs_wait();

                            // For RM: always transpose back so the packed tile is in
                            // regular (non-transposed) format, which pack_untilize_block
                            // expects.  For TILE: only do the transpose-back at the last
                            // stage/sub (intermediate stages keep the transposed format in
                            // DRAM to avoid an extra transpose next stage).
                            if ((stage == log2Wt && sub == 1) || is_row_major) {
                                cb_reserve_back(input_tensor_transposed_cb_index, 2 * one_tile);
                                cb_reserve_back(index_tensor_transposed_cb_index, 2 * one_tile);

                                pack_reconfig_data_format(input_tensor_transposed_cb_index);
                                pack_tile(tile_input_low, input_tensor_transposed_cb_index);
                                pack_tile(tile_input_high, input_tensor_transposed_cb_index);

                                pack_reconfig_data_format(index_tensor_transposed_cb_index);
                                pack_tile(tile_index_low, index_tensor_transposed_cb_index);
                                pack_tile(tile_index_high, index_tensor_transposed_cb_index);

                                cb_push_back(input_tensor_transposed_cb_index, 2 * one_tile);
                                cb_push_back(index_tensor_transposed_cb_index, 2 * one_tile);

                                tile_regs_release();

                                acquire_dst();

                                cb_wait_front(input_tensor_transposed_cb_index, 2 * one_tile);
                                reconfig_data_format_srca(input_tensor_transposed_cb_index);
                                transpose_wh_init_short(input_tensor_transposed_cb_index);
                                transpose_wh_tile(input_tensor_transposed_cb_index, 0, input_dest_start);
                                transpose_wh_tile(input_tensor_transposed_cb_index, 1, input_dest_end);

                                cb_reserve_back(input_tensor_output_cb_index, 2 * one_tile);

                                // Process value tiles
                                pack_reconfig_data_format(input_tensor_output_cb_index);
                                pack_tile(input_dest_start, input_tensor_output_cb_index);
                                pack_tile(input_dest_end, input_tensor_output_cb_index);

                                cb_pop_front(input_tensor_transposed_cb_index, 2 * one_tile);
                                // Push tiles to writer
                                cb_push_back(input_tensor_output_cb_index, 2 * one_tile);

                                release_dst();

                                acquire_dst();

                                cb_wait_front(index_tensor_transposed_cb_index, 2 * one_tile);
                                reconfig_data_format_srca(index_tensor_transposed_cb_index);
                                transpose_wh_init_short(index_tensor_transposed_cb_index);
                                transpose_wh_tile(index_tensor_transposed_cb_index, 0, input_dest_start);
                                transpose_wh_tile(index_tensor_transposed_cb_index, 1, input_dest_end);

                                cb_reserve_back(index_tensor_output_cb_index, 2 * one_tile);
                                pack_reconfig_data_format(index_tensor_output_cb_index);
                                pack_tile(input_dest_start, index_tensor_output_cb_index);
                                pack_tile(input_dest_end, index_tensor_output_cb_index);

                                cb_pop_front(index_tensor_transposed_cb_index, 2 * one_tile);
                                cb_push_back(index_tensor_output_cb_index, 2 * one_tile);

                                release_dst();
                            } else {
                                // Intermediate step - pack and push transposed tiles to be saved for the next stage
                                cb_reserve_back(index_tensor_output_cb_index, 2 * one_tile);
                                cb_reserve_back(input_tensor_output_cb_index, 2 * one_tile);

                                // Process value tiles
                                pack_reconfig_data_format(input_tensor_output_cb_index);
                                pack_tile(tile_input_low, input_tensor_output_cb_index);
                                pack_tile(tile_input_high, input_tensor_output_cb_index);

                                pack_reconfig_data_format(index_tensor_output_cb_index);
                                pack_tile(tile_index_low, index_tensor_output_cb_index);
                                pack_tile(tile_index_high, index_tensor_output_cb_index);

                                cb_push_back(input_tensor_output_cb_index, 2 * one_tile);
                                cb_push_back(index_tensor_output_cb_index, 2 * one_tile);

                                tile_regs_release();
                            }

                            if constexpr (is_row_major) {
                                pack_untilize_init<false>(input_tensor_output_cb_index, 2, rm_output_value_cb_index);
                                cb_wait_front(input_tensor_output_cb_index, 2);
                                cb_reserve_back(rm_output_value_cb_index, 2 * TILE_H);
                                pack_untilize_block<false>(input_tensor_output_cb_index, 2, rm_output_value_cb_index);
                                cb_pop_front(input_tensor_output_cb_index, 2);
                                cb_push_back(rm_output_value_cb_index, 2 * TILE_H);
                                binary_op_init_common(
                                    rm_input_index_cb_index, rm_input_index_cb_index, index_tensor_output_cb_index);

                                pack_untilize_init<false>(index_tensor_output_cb_index, 2, rm_output_index_cb_index);
                                cb_wait_front(index_tensor_output_cb_index, 2);
                                cb_reserve_back(rm_output_index_cb_index, 2 * TILE_H);
                                pack_untilize_block<false>(index_tensor_output_cb_index, 2, rm_output_index_cb_index);
                                cb_pop_front(index_tensor_output_cb_index, 2);
                                cb_push_back(rm_output_index_cb_index, 2 * TILE_H);
                                // Reset compute state for the next pair's tilize.
                                binary_op_init_common(
                                    rm_input_value_cb_index, rm_input_index_cb_index, input_tensor_cb_index);
                            }

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
