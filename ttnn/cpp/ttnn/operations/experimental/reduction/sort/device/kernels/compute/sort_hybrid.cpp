// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/binary_max_min.h"

#include "sort_common.hpp"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(5);
    constexpr bool ascending = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(8);
    constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(9);
    constexpr uint32_t index_tensor_transposed_cb_index = get_compile_time_arg_val(10);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(11);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(12);
    constexpr uint32_t value_tensor_intermediate_cb_index = get_compile_time_arg_val(13);
    constexpr uint32_t index_tensor_intermediate_cb_index = get_compile_time_arg_val(14);
    constexpr uint32_t value_tensor_peer_cb_index = get_compile_time_arg_val(15);
    constexpr uint32_t index_tensor_peer_cb_index = get_compile_time_arg_val(16);
    constexpr uint32_t packer_unpacker_sync_cb_index = get_compile_time_arg_val(17);

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

    // LLK setup
    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        bool dir = ascending ^ ((core_id & 1) == 1);

        // Read input value data
        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_cb_index,
            index_tensor_cb_index,
            input_tensor_transposed_cb_index,
            index_tensor_transposed_cb_index,
            number_of_tiles_per_core,
            /*switch_dir=*/true,
            dir,
            /*end_phase(log2(K))=*/5);

        global_old_cb = index_tensor_cb_index;

        // Wait for bitonic sequence of Wt tiles
        cb_wait_front(input_tensor_transposed_cb_index, number_of_tiles_per_core);
        cb_wait_front(index_tensor_transposed_cb_index, number_of_tiles_per_core);

        // Sort and merge step of bitonic merge sort
        uint32_t stages = ilog2(Wt);
        for (uint32_t stage = 2; stage <= stages; stage++) {
            const uint32_t m_iter = stage - 1;

            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                uint16_t pair_id = 0;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    // Tile i not on this core - nothing to do
                    if (i < global_tile_start || i >= global_tile_end) {
                        continue;
                    }

                    sync_packer_unpacker(packer_unpacker_sync_cb_index);

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
                            copy_tile_to_dst_init_with_cb_update(input_tensor_transposed_cb_index, global_old_cb);
                            copy_tile(input_tensor_transposed_cb_index, left_tile_id, input_dest_start);
                            copy_tile(input_tensor_transposed_cb_index, right_tile_id, input_dest_end);

                            // Copy index tiles to DST register
                            copy_tile_to_dst_init_with_cb_update(index_tensor_transposed_cb_index, global_old_cb);
                            copy_tile(index_tensor_transposed_cb_index, left_tile_id, index_dest_start);
                            copy_tile(index_tensor_transposed_cb_index, right_tile_id, index_dest_end);

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
                            pack_reconfig_data_format(input_tensor_transposed_cb_index);
                            pack_tile<true>(tile_input_low, input_tensor_transposed_cb_index, left_tile_id);
                            pack_tile<true>(tile_input_high, input_tensor_transposed_cb_index, right_tile_id);

                            // Pack index tiles to CB
                            pack_reconfig_data_format(index_tensor_transposed_cb_index);
                            pack_tile<true>(tile_index_low, index_tensor_transposed_cb_index, left_tile_id);
                            pack_tile<true>(tile_index_high, index_tensor_transposed_cb_index, right_tile_id);

                            tile_regs_release();
                        }
                    } else {
                        const uint32_t tile_id = i - global_tile_start;
                        constexpr uint32_t FIRST_TILE = 0;

                        // Send tiles to other core
                        tile_regs_acquire();

                        // Copy index tiles to DST register for exchange
                        copy_tile_to_dst_init_with_cb_update(index_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(index_tensor_transposed_cb_index, tile_id, index_dest_start);

                        // Copy value tiles to DST register for exchange
                        copy_tile_to_dst_init_with_cb_update(input_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(input_tensor_transposed_cb_index, tile_id, input_dest_start);

                        tile_regs_commit();
                        tile_regs_wait();

                        // Send current index tile reader for exchange
                        cb_reserve_back(index_tensor_intermediate_cb_index, one_tile);
                        pack_reconfig_data_format(index_tensor_intermediate_cb_index);
                        pack_tile(index_dest_start, index_tensor_intermediate_cb_index, FIRST_TILE);
                        cb_push_back(index_tensor_intermediate_cb_index, one_tile);

                        // Send current value tile reader for exchange
                        cb_reserve_back(value_tensor_intermediate_cb_index, one_tile);
                        pack_reconfig_data_format(value_tensor_intermediate_cb_index);
                        pack_tile(input_dest_start, value_tensor_intermediate_cb_index, FIRST_TILE);
                        cb_push_back(value_tensor_intermediate_cb_index, one_tile);

                        tile_regs_release();

                        sync_packer_unpacker(packer_unpacker_sync_cb_index);

                        // Process received tiles from other core
                        tile_regs_acquire();

                        // Prepare local index tiles for sorting with new tiles
                        copy_tile_to_dst_init_with_cb_update(index_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(index_tensor_transposed_cb_index, tile_id, index_dest_start);

                        // Prepare local value tiles for sorting with new tiles
                        copy_tile_to_dst_init_with_cb_update(input_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(input_tensor_transposed_cb_index, tile_id, input_dest_start);

                        cb_wait_front(index_tensor_peer_cb_index, one_tile);

                        // Load new index tile for sorting
                        copy_tile_to_dst_init_with_cb_update(index_tensor_peer_cb_index, global_old_cb);
                        copy_tile(index_tensor_peer_cb_index, FIRST_TILE, index_dest_end);

                        cb_pop_front(index_tensor_peer_cb_index, one_tile);

                        // Read other tile from writer
                        cb_wait_front(value_tensor_peer_cb_index, one_tile);

                        // Load new value tile for sorting
                        copy_tile_to_dst_init_with_cb_update(value_tensor_peer_cb_index, global_old_cb);
                        copy_tile(value_tensor_peer_cb_index, FIRST_TILE, input_dest_end);

                        cb_pop_front(value_tensor_peer_cb_index, one_tile);

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
                        pack_reconfig_data_format(index_tensor_transposed_cb_index);
                        pack_tile<true>(index_output_tile, index_tensor_transposed_cb_index, tile_id);

                        // Pack sorted value tiles to CB
                        pack_reconfig_data_format(input_tensor_transposed_cb_index);
                        pack_tile<true>(value_output_tile, input_tensor_transposed_cb_index, tile_id);

                        tile_regs_release();
                    }
                }  // Wt loop
            }  // sub loop
        }  // stages loop

        cb_reserve_back(input_tensor_transposed_cb_index, number_of_tiles_per_core);
        cb_reserve_back(index_tensor_transposed_cb_index, number_of_tiles_per_core);

        cb_pop_front(input_tensor_transposed_cb_index, number_of_tiles_per_core);
        cb_pop_front(index_tensor_transposed_cb_index, number_of_tiles_per_core);

        cb_push_back(input_tensor_transposed_cb_index, number_of_tiles_per_core);
        cb_push_back(index_tensor_transposed_cb_index, number_of_tiles_per_core);

        // Values tensor
        transpose_and_pack(input_tensor_transposed_cb_index, value_tensor_cb_index, number_of_tiles_per_core);

        // Indexes tensor
        transpose_and_pack(index_tensor_transposed_cb_index, index_tensor_output_cb_index, number_of_tiles_per_core);
    }  // h loop
}  // MAIN
}  // namespace NAMESPACE
