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
#include "api/compute/binary_max_min.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

#include "sort_common.hpp"

#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(5);
    constexpr bool ascending = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t index_tensor_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t input_tensor_transposed_cb_id = get_compile_time_arg_val(9);
    constexpr uint32_t index_tensor_transposed_cb_id = get_compile_time_arg_val(10);
    constexpr uint32_t value_tensor_cb_id = get_compile_time_arg_val(11);
    constexpr uint32_t index_tensor_output_cb_id = get_compile_time_arg_val(12);
    constexpr uint32_t value_tensor_intermediate_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t index_tensor_intermediate_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t value_tensor_peer_cb_id = get_compile_time_arg_val(15);
    constexpr uint32_t index_tensor_peer_cb_id = get_compile_time_arg_val(16);
    constexpr uint32_t packer_unpacker_sync_cb_id = get_compile_time_arg_val(17);
    constexpr bool is_row_major = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t rm_input_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t rm_value_output_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t rm_index_output_cb_id = get_compile_time_arg_val(21);
    constexpr uint32_t rm_post_sort_index_cb_id = get_compile_time_arg_val(22);

    CircularBuffer input_tensor_cb(input_tensor_cb_id);
    CircularBuffer index_tensor_cb(index_tensor_cb_id);
    CircularBuffer input_tensor_transposed_cb(input_tensor_transposed_cb_id);
    CircularBuffer index_tensor_transposed_cb(index_tensor_transposed_cb_id);
    CircularBuffer value_tensor_cb(value_tensor_cb_id);
    CircularBuffer index_tensor_output_cb(index_tensor_output_cb_id);
    CircularBuffer value_tensor_intermediate_cb(value_tensor_intermediate_cb_id);
    CircularBuffer index_tensor_intermediate_cb(index_tensor_intermediate_cb_id);
    CircularBuffer value_tensor_peer_cb(value_tensor_peer_cb_id);
    CircularBuffer index_tensor_peer_cb(index_tensor_peer_cb_id);
    CircularBuffer packer_unpacker_sync_cb(packer_unpacker_sync_cb_id);
    CircularBuffer rm_input_cb(rm_input_cb_id);
    CircularBuffer rm_value_output_cb(rm_value_output_cb_id);
    CircularBuffer rm_index_output_cb(rm_index_output_cb_id);
    CircularBuffer rm_post_sort_index_cb(rm_post_sort_index_cb_id);

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_pair_start = core_id * number_of_pairs_processed_by_each_core;
    const uint16_t processing_pair_end = processing_pair_start + number_of_pairs_processed_by_each_core;

    uint32_t global_old_cb = 0;

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    // LLK setup — one compute_kernel_hw_startup at the start, then full inits.
    compute_kernel_hw_startup(
        is_row_major ? rm_input_cb_id : input_tensor_cb_id, index_tensor_cb_id, input_tensor_cb_id);
    if constexpr (is_row_major) {
        binary_op_init_common(input_tensor_cb_id, index_tensor_cb_id, input_tensor_transposed_cb_id);
    }
    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_id, input_tensor_transposed_cb_id);

    for (uint32_t h = 0; h < Ht; h++) {
        if constexpr (is_row_major) {
            constexpr uint32_t TILE_H = 32;
            tilize_init(rm_input_cb_id, number_of_tiles_per_core, input_tensor_cb_id);
            rm_input_cb.wait_front(TILE_H);
            input_tensor_cb.reserve_back(number_of_tiles_per_core);
            tilize_block(rm_input_cb_id, number_of_tiles_per_core, input_tensor_cb_id);
            input_tensor_cb.push_back(number_of_tiles_per_core);
            rm_input_cb.pop_front(TILE_H);
            tilize_uninit(rm_input_cb_id, input_tensor_cb_id);
        }

        bool dir = ascending ^ ((core_id & 1) == 1);

        // Read input value data
        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_cb,
            index_tensor_cb,
            input_tensor_transposed_cb,
            index_tensor_transposed_cb,
            number_of_tiles_per_core,
            /*switch_dir=*/true,
            dir,
            /*end_phase(log2(K))=*/5);

        global_old_cb = index_tensor_cb_id;

        // Wait for bitonic sequence of Wt tiles
        input_tensor_transposed_cb.wait_front(number_of_tiles_per_core);
        index_tensor_transposed_cb.wait_front(number_of_tiles_per_core);

        // Sort and merge step of bitonic merge sort
        const uint32_t stages = ilog2(Wt);
        for (uint32_t stage = 2; stage <= stages; stage++) {
            const uint32_t m_iter = stage - 1;

            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    // Tile i not on this core - nothing to do
                    if (i < global_tile_start || i >= global_tile_end) {
                        continue;
                    }

                    sync_packer_unpacker(packer_unpacker_sync_cb);

                    // Determine direction for this comparison block
                    const bool ascending_block = ((i >> stage) & 1) == 0;
                    const bool dir = ascending_block == ascending;

                    if (j >= global_tile_start && j < global_tile_end) {
                        if (j > i) {
                            // Local sorting - both tiles in core memory

                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i - global_tile_start;
                            const uint32_t right_tile_id = j - global_tile_start;

                            tile_regs_acquire();

                            // Copy value tiles to DST register
                            copy_tile_to_dst_init_with_cb_update(input_tensor_transposed_cb_id, global_old_cb);
                            copy_tile(input_tensor_transposed_cb_id, left_tile_id, input_dest_start);
                            copy_tile(input_tensor_transposed_cb_id, right_tile_id, input_dest_end);

                            // Copy index tiles to DST register
                            copy_tile_to_dst_init_with_cb_update(index_tensor_transposed_cb_id, global_old_cb);
                            copy_tile(index_tensor_transposed_cb_id, left_tile_id, index_dest_start);
                            copy_tile(index_tensor_transposed_cb_id, right_tile_id, index_dest_end);

                            uint32_t tile_input_low = input_dest_start;
                            uint32_t tile_input_high = input_dest_end;
                            uint32_t tile_index_low = index_dest_start;
                            uint32_t tile_index_high = index_dest_end;

                            if (sub == 1) {
                                // Use sort LLK only the last stage to sort the last pair of tiles - speed up
                                ckernel::topk_local_sort(/*idst=*/0, (int)dir, /*end_phase(log2(K))=*/5);
                            } else {
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

                            // Pack value tiles to CB
                            pack_reconfig_data_format(input_tensor_transposed_cb_id);
                            pack_tile<true>(tile_input_low, input_tensor_transposed_cb_id, left_tile_id);
                            pack_tile<true>(tile_input_high, input_tensor_transposed_cb_id, right_tile_id);

                            // Pack index tiles to CB
                            pack_reconfig_data_format(index_tensor_transposed_cb_id);
                            pack_tile<true>(tile_index_low, index_tensor_transposed_cb_id, left_tile_id);
                            pack_tile<true>(tile_index_high, index_tensor_transposed_cb_id, right_tile_id);

                            tile_regs_release();
                        }
                    } else {
                        const uint32_t tile_id = i - global_tile_start;
                        constexpr uint32_t FIRST_TILE = 0;

                        if ((i & 1) == 0) {  // i % 2
                            value_tensor_intermediate_cb.reserve_back(one_tile);
                            index_tensor_intermediate_cb.reserve_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb, index_tensor_transposed_cb, tile_id, index_tensor_intermediate_cb);
                            index_tensor_intermediate_cb.push_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb, input_tensor_transposed_cb, tile_id, value_tensor_intermediate_cb);
                            value_tensor_intermediate_cb.push_back(one_tile);

                            value_tensor_intermediate_cb.reserve_back(one_tile);
                            index_tensor_intermediate_cb.reserve_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb, index_tensor_transposed_cb, tile_id + 1, index_tensor_intermediate_cb);
                            index_tensor_intermediate_cb.push_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb, input_tensor_transposed_cb, tile_id + 1, value_tensor_intermediate_cb);
                            value_tensor_intermediate_cb.push_back(one_tile);
                            sync_packer_unpacker(packer_unpacker_sync_cb);
                        }

                        // Process received tiles from other core
                        tile_regs_acquire();

                        // Prepare local index tiles for sorting with new tiles
                        copy_tile_to_dst_init_with_cb_update(index_tensor_transposed_cb_id, global_old_cb);
                        copy_tile(index_tensor_transposed_cb_id, tile_id, index_dest_start);

                        // Prepare local value tiles for sorting with new tiles
                        copy_tile_to_dst_init_with_cb_update(input_tensor_transposed_cb_id, global_old_cb);
                        copy_tile(input_tensor_transposed_cb_id, tile_id, input_dest_start);

                        index_tensor_peer_cb.wait_front(one_tile);

                        // Load new index tile for sorting
                        copy_tile_to_dst_init_with_cb_update(index_tensor_peer_cb_id, global_old_cb);
                        copy_tile(index_tensor_peer_cb_id, FIRST_TILE, index_dest_end);

                        index_tensor_peer_cb.pop_front(one_tile);

                        // Read other tile from writer
                        value_tensor_peer_cb.wait_front(one_tile);

                        // Load new value tile for sorting
                        copy_tile_to_dst_init_with_cb_update(value_tensor_peer_cb_id, global_old_cb);
                        copy_tile(value_tensor_peer_cb_id, FIRST_TILE, input_dest_end);

                        value_tensor_peer_cb.pop_front(one_tile);

                        ckernel::topk_merge(0, m_iter, 32);

                        // topk_merge puts smallest values in DEST[0] and largest in DEST[1]
                        // If core must keep smallest values, then keep DEST[1] instead of DEST[0]
                        const uint32_t select_lower = dir ^ (i < j);

                        uint32_t value_output_tile = input_dest_start;
                        uint32_t index_output_tile = index_dest_start;
                        if (!select_lower) {
                            value_output_tile = input_dest_end;
                            index_output_tile = index_dest_end;
                        }

                        tile_regs_commit();
                        tile_regs_wait();

                        // Pack sorted index tiles to CB
                        pack_reconfig_data_format(index_tensor_transposed_cb_id);
                        pack_tile<true>(index_output_tile, index_tensor_transposed_cb_id, tile_id);

                        // Pack sorted value tiles to CB
                        pack_reconfig_data_format(input_tensor_transposed_cb_id);
                        pack_tile<true>(value_output_tile, input_tensor_transposed_cb_id, tile_id);

                        tile_regs_release();
                    }
                }  // Wt loop
            }  // sub loop
        }  // stages loop

        input_tensor_transposed_cb.reserve_back(number_of_tiles_per_core);
        index_tensor_transposed_cb.reserve_back(number_of_tiles_per_core);

        input_tensor_transposed_cb.pop_front(number_of_tiles_per_core);
        index_tensor_transposed_cb.pop_front(number_of_tiles_per_core);

        input_tensor_transposed_cb.push_back(number_of_tiles_per_core);
        index_tensor_transposed_cb.push_back(number_of_tiles_per_core);

        if constexpr (!is_row_major) {
            transpose_and_pack(input_tensor_transposed_cb, value_tensor_cb, number_of_tiles_per_core);
            transpose_and_pack(index_tensor_transposed_cb, index_tensor_output_cb, number_of_tiles_per_core);
        } else {
            // ROW_MAJOR output: un-transpose the sorted tiles back into the
            // PACK-only RM-input/index CBs (which are now empty after the
            // tilize_block/sort loops drained them), then pack_untilize them
            // into TILE_H RM rows for the writer/reader to drain.
            constexpr uint32_t TILE_H = 32;
            // DST_ACCUM_MODE is a compile-time macro injected by the framework when
            // fp32_dest_acc_en=true is set in ComputeConfigDescriptor (controlled by
            // is_32_bit_data in sort_program_factory.cpp: true for Float32 input or
            // UInt32 index).  MAX_DEST_TILES is therefore data-format-dependent:
            // 32-bit DEST holds 4 tiles; 16-bit (BF16) DEST holds 8 tiles.
            constexpr uint32_t MAX_DEST_TILES = DST_ACCUM_MODE ? 4 : 8;
            // number_of_tiles_per_core is a power-of-two: get_number_of_tiles_per_core()
            // returns Wt / num_cores, and Wt is always a power-of-two (padded by
            // pre_sort_transform_tensor).  MAX_DEST_TILES is also a power-of-two (4 or 8),
            // so number_of_tiles_per_core % SUB_BLOCK_DIM == 0 is always satisfied.
            constexpr uint32_t SUB_BLOCK_DIM =
                (number_of_tiles_per_core < MAX_DEST_TILES) ? number_of_tiles_per_core : MAX_DEST_TILES;
            constexpr uint32_t NUM_SUB_BLOCKS = number_of_tiles_per_core / SUB_BLOCK_DIM;
            static_assert(
                number_of_tiles_per_core % SUB_BLOCK_DIM == 0,
                "number_of_tiles_per_core must be divisible by SUB_BLOCK_DIM");

            transpose_and_pack(input_tensor_transposed_cb, input_tensor_cb, number_of_tiles_per_core);

            transpose_and_pack(index_tensor_transposed_cb, rm_post_sort_index_cb, number_of_tiles_per_core);

            // Untilize values: number_of_tiles_per_core tiles → TILE_H RM pages.
            binary_op_init_common(input_tensor_cb_id, index_tensor_cb_id, rm_value_output_cb_id);
            pack_untilize_init<SUB_BLOCK_DIM, number_of_tiles_per_core>(input_tensor_cb_id, rm_value_output_cb_id);
            input_tensor_cb.wait_front(number_of_tiles_per_core);
            rm_value_output_cb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, number_of_tiles_per_core>(
                    input_tensor_cb_id, 1, rm_value_output_cb_id, b);
                input_tensor_cb.pop_front(SUB_BLOCK_DIM);
            }
            rm_value_output_cb.push_back(TILE_H);
            pack_untilize_uninit(rm_value_output_cb_id);

            // Untilize indices: number_of_tiles_per_core tiles → TILE_H RM pages.
            binary_op_init_common(rm_post_sort_index_cb_id, input_tensor_cb_id, rm_index_output_cb_id);
            pack_untilize_init<SUB_BLOCK_DIM, number_of_tiles_per_core>(
                rm_post_sort_index_cb_id, rm_index_output_cb_id);
            rm_post_sort_index_cb.wait_front(number_of_tiles_per_core);
            rm_index_output_cb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, number_of_tiles_per_core>(
                    rm_post_sort_index_cb_id, 1, rm_index_output_cb_id, b);
                rm_post_sort_index_cb.pop_front(SUB_BLOCK_DIM);
            }
            rm_index_output_cb.push_back(TILE_H);
            pack_untilize_uninit(rm_index_output_cb_id);
        }
    }  // h loop
}  // void kernel_main()
