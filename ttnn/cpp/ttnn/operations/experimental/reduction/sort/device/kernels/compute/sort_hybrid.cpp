// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"

#include "debug/dprint.h"
#include "debug/waypoint.h"
#include "debug/pause.h"
#include "debug/dprint_tensix.h"

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

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_pair_start = core_id * number_of_pairs_processed_by_each_core;
    const uint16_t processing_pair_end = processing_pair_start + number_of_pairs_processed_by_each_core;

    DPRINT << "COMPUTE: "
           << " input_cb_index: " << input_tensor_cb_index << " index_output_cb_index: " << index_tensor_output_cb_index
           << " value_intermediate = " << value_tensor_intermediate_cb_index
           << " index_intermediate = " << index_tensor_intermediate_cb_index
           << " value_peer_cb_index: " << value_tensor_peer_cb_index
           << " index_peer_cb_index: " << index_tensor_peer_cb_index << " Ht: " << Ht << " Wt: " << Wt
           << " number_of_tiles_per_core: " << number_of_tiles_per_core
           << " number_of_cores_used: " << number_of_cores_used << " ascending: " << (uint32_t)ascending << ENDL();

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        // PAUSE();  // TODO: Remove
        // Create input sequence
        DPRINT << "COMPUTE: Generating bitonic sequence" << ENDL();

        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_cb_index,
            index_tensor_cb_index,
            input_tensor_transposed_cb_index,
            index_tensor_transposed_cb_index,
            number_of_tiles_per_core,
            /*switch_dir=*/true,
            ascending,
            /*end_phase(log2(K))=*/5);

        // Wait for bitonic sequence of Wt tiles
        cb_wait_front(input_tensor_transposed_cb_index, number_of_tiles_per_core);
        cb_wait_front(index_tensor_transposed_cb_index, number_of_tiles_per_core);

        DPRINT << "COMPUTE: Bitonic sequence OK" << ENDL();

        // Sort and merge step of bitonic merge sort
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                uint16_t pair_id = 0;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    // Tile i not on this core
                    if (i < global_tile_start || i >= global_tile_end) {
                        continue;
                    }

                    // Determine direction for this comparison block
                    const bool ascending_block = ((i >> stage) & 1) == 0;
                    const bool dir = ascending_block == ascending;

                    if (j >= global_tile_start && j < global_tile_end) {
                        if (j > i) {
                            // Local sorting

                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i - global_tile_start;
                            const uint32_t right_tile_id = j - global_tile_start;
                            // PAUSE();  // TODO: Remove
                            tile_regs_acquire();

                            copy_tile_to_dst_init_short_with_dt(
                                input_tensor_transposed_cb_index, index_tensor_transposed_cb_index);
                            copy_tile(index_tensor_transposed_cb_index, left_tile_id, index_dest_start);
                            copy_tile(index_tensor_transposed_cb_index, right_tile_id, index_dest_end);

                            copy_tile_to_dst_init_short_with_dt(
                                index_tensor_transposed_cb_index, input_tensor_transposed_cb_index);
                            copy_tile(input_tensor_transposed_cb_index, left_tile_id, input_dest_start);
                            copy_tile(input_tensor_transposed_cb_index, right_tile_id, input_dest_end);

                            ckernel::topk_local_sort(0, (int)dir, 5);

                            tile_regs_commit();
                            tile_regs_wait();

                            pack_reconfig_data_format(input_tensor_transposed_cb_index);
                            pack_tile<true>(input_dest_start, input_tensor_transposed_cb_index, left_tile_id);
                            pack_tile<true>(input_dest_end, input_tensor_transposed_cb_index, right_tile_id);

                            pack_reconfig_data_format(index_tensor_transposed_cb_index);
                            pack_tile<true>(index_dest_start, index_tensor_transposed_cb_index, left_tile_id);
                            pack_tile<true>(index_dest_end, index_tensor_transposed_cb_index, right_tile_id);

                            tile_regs_release();
                        }
                    } else {
                        DPRINT << "COMPUTE: exchanging tiles " << i << " and " << j << ENDL();

                        const uint32_t tile_id = 0;  // TODO: Compute correct index
                        constexpr uint32_t FIRST_TILE = 0;

                        tile_regs_acquire();
                        // Read from
                        copy_tile_to_dst_init_short_with_dt(
                            input_tensor_transposed_cb_index, index_tensor_transposed_cb_index);
                        copy_tile(index_tensor_transposed_cb_index, tile_id, index_dest_start);

                        copy_tile_to_dst_init_short_with_dt(
                            index_tensor_transposed_cb_index, input_tensor_transposed_cb_index);
                        copy_tile(input_tensor_transposed_cb_index, tile_id, input_dest_start);

                        DPRINT_MATH(DPRINT << "THIS INPUT [input_dest_start] = " << ENDL());
                        dprint_tensix_dest_reg(input_dest_start);

                        tile_regs_commit();

                        tile_regs_wait();

                        // Send current index to reader
                        cb_reserve_back(index_tensor_intermediate_cb_index, one_tile);
                        pack_reconfig_data_format(index_tensor_intermediate_cb_index);
                        pack_tile<true>(index_dest_start, index_tensor_intermediate_cb_index, FIRST_TILE);
                        cb_push_back(index_tensor_intermediate_cb_index, one_tile);

                        // Send current tile to writer
                        cb_reserve_back(value_tensor_intermediate_cb_index, one_tile);
                        pack_reconfig_data_format(value_tensor_intermediate_cb_index);
                        pack_tile<true>(input_dest_start, value_tensor_intermediate_cb_index, FIRST_TILE);
                        cb_push_back(value_tensor_intermediate_cb_index, one_tile);

                        tile_regs_release();

                        // TODO: Do we need to Sync Unpacker/Packer ?
                        //       It may not be necessary because dest regs values are different

                        // Read other indices from reader
                        tile_regs_acquire();
                        cb_wait_front(index_tensor_peer_cb_index, one_tile);
                        copy_tile_to_dst_init_short_with_dt(
                            input_tensor_transposed_cb_index, index_tensor_peer_cb_index);
                        copy_tile(index_tensor_peer_cb_index, FIRST_TILE, index_dest_end);
                        cb_pop_front(index_tensor_peer_cb_index, one_tile);

                        // Read other tile from writer
                        cb_wait_front(value_tensor_peer_cb_index, one_tile);
                        copy_tile_to_dst_init_short_with_dt(index_tensor_peer_cb_index, value_tensor_peer_cb_index);
                        copy_tile(value_tensor_peer_cb_index, FIRST_TILE, input_dest_end);
                        cb_pop_front(value_tensor_peer_cb_index, one_tile);

                        DPRINT_MATH(DPRINT << "INPUT0 [input_dest_start] = " << ENDL());
                        dprint_tensix_dest_reg(input_dest_start);

                        DPRINT_MATH(DPRINT << "INPUT1 [input_dest_end] = " << ENDL());
                        dprint_tensix_dest_reg(input_dest_end);

                        ckernel::topk_local_sort(0, (int)dir, 5);

                        // TODO: Fix output tile selection w.r.t ascending
                        uint32_t value_output_tile = input_dest_start;
                        uint32_t index_output_tile = index_dest_start;
                        if (i > j) {
                            value_output_tile = input_dest_end;
                            index_output_tile = index_dest_end;
                        }

                        DPRINT << "OUTPUT [value_output_tile] = ";
                        dprint_tensix_dest_reg(value_output_tile);

                        tile_regs_commit();

                        tile_regs_wait();
                        pack_reconfig_data_format(input_tensor_transposed_cb_index);
                        pack_tile<true>(value_output_tile, input_tensor_transposed_cb_index, tile_id);
                        // pack_tile<true>(input_dest_end, input_tensor_transposed_cb_index, right_tile_id);

                        pack_reconfig_data_format(index_tensor_transposed_cb_index);
                        pack_tile<true>(index_output_tile, index_tensor_transposed_cb_index, tile_id);
                        // pack_tile<true>(index_dest_end, index_tensor_transposed_cb_index, right_tile_id);
                        tile_regs_release();

                        // TODO: Sync Unpacker/Packer
                        //       Otherwise, unpacker could start next iteration while packer is not done yet
                        //       Note: This may be a problem for both conditional
                    }
                }  // Wt
            }  // sub
        }  // stages
        DPRINT << "COMPUTE: AFTER LOGIC" << ENDL();

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
    DPRINT << "COMPUTE: Finished reading and sorting tiles." << ENDL();  // TODO: Remove
}  // MAIN
}  // namespace NAMESPACE
