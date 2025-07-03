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

#include "debug/dprint.h"
#include "debug/waypoint.h"
#include "debug/pause.h"
#include "debug/dprint_tensix.h"

#include "sort_common.hpp"

#include <cstdint>

FORCE_INLINE
void COPY_TILE_TO_DST_INIT(uint32_t new_cb, uint32_t& global_old_cb) {
    // copy_tile_to_dst_init_short(new_cb);

    copy_tile_to_dst_init_short_with_dt(global_old_cb, new_cb);
    global_old_cb = new_cb;
}

FORCE_INLINE
void print_Wt_tiles(uint32_t cb_value, uint32_t Wt) {
    // Debug: Print data

    // tile_regs_wait();
    // // Nothing to do: We are only printing tiles
    // pack_reconfig_data_format(cb_value);
    // // pack_tile(INPUT_TILE0, cb_value, w);

    // tile_regs_release();

    for (uint32_t w = 0; w < Wt; w++) {
        constexpr uint32_t INPUT_TILE0 = 0;

        DPRINT << "[" << w << "]" << ENDL();
        // input_tensor_transposed_cb_index already in reserved state
        tile_regs_acquire();
        // reconfig_data_format_srca(cb_value);
        copy_tile_to_dst_init_short(cb_value);
        copy_tile(cb_value, w, INPUT_TILE0);

        // binary_min_tile_init();
        // binary_min_tile(INPUT_TILE0, INPUT_TILE0);

        dprint_tensix_dest_reg(INPUT_TILE0);
        tile_regs_commit();

        tile_regs_wait();
        tile_regs_release();
    }
}

constexpr uint32_t ilog2(uint32_t n) { return 31 - __builtin_clz(n); }

FORCE_INLINE
void sync_packer_unpacker(uint32_t packer_unpacker_sync_cb_index) {
    constexpr uint32_t ONE_TILE = 1;
    // MATH(WAYPOINT("SPU"));

    // DPRINT << "COMPUTE: #1 at " << __LINE__ << ENDL();
    cb_reserve_back(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_push_back(packer_unpacker_sync_cb_index, ONE_TILE);

    // DPRINT << "COMPUTE: #2 at " << __LINE__ << ENDL();
    cb_wait_front(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_pop_front(packer_unpacker_sync_cb_index, ONE_TILE);

    // DPRINT << "COMPUTE: #3 at " << __LINE__ << ENDL();
    cb_reserve_back(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_push_back(packer_unpacker_sync_cb_index, ONE_TILE);

    // DPRINT << "COMPUTE: #4 at " << __LINE__ << ENDL();
    cb_wait_front(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_pop_front(packer_unpacker_sync_cb_index, ONE_TILE);

    // MATH(WAYPOINT("SPUD"));
}

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

    // DPRINT << "COMPUTE: "
    //        << " input_cb_index: " << input_tensor_cb_index << " index_output_cb_index: " <<
    //        index_tensor_output_cb_index
    //        << " value_intermediate = " << value_tensor_intermediate_cb_index
    //        << " index_intermediate = " << index_tensor_intermediate_cb_index
    //        << " value_peer_cb_index: " << value_tensor_peer_cb_index
    //        << " index_peer_cb_index: " << index_tensor_peer_cb_index << " Ht: " << Ht << " Wt: " << Wt
    //        << " number_of_tiles_per_core: " << number_of_tiles_per_core
    //        << " number_of_cores_used: " << number_of_cores_used << " ascending: " << (uint32_t)ascending << ENDL();

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    // 'Fill' packer_unpacker_sync_cb_index
    // In our case, we want to sometime make packer wait for unpacker (i.e. for DPRINT)
    // cb_reserve_back(packer_unpacker_sync_cb_index, one_tile);
    // cb_push_back(packer_unpacker_sync_cb_index, one_tile);

    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        // PAUSE();  // TODO: Remove
        // Create input sequence
        // DPRINT << "COMPUTE: Generatinag bitonic sequence" << ENDL();

        bool dir = ascending ^ ((core_id & 1) == 1);
        DPRINT << "COMPUTE: core_id = " << core_id << ", ascending = " << (uint32_t)ascending
               << ", dir = " << (uint32_t)dir << ENDL();

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
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }
        for (uint32_t stage = 2; stage <= stages; stage++) {
            const uint32_t m_iter = stage - 1;  // used as topk_merge / topk_rebuild argument

            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                uint16_t pair_id = 0;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    sync_packer_unpacker(packer_unpacker_sync_cb_index);

                    // Tile i not on this core
                    if (i < global_tile_start || i >= global_tile_end) {
                        continue;
                    }

                    // Determine direction for this comparison block
                    const bool ascending_block = ((i >> stage) & 1) == 0;
                    const bool dir = ascending_block == ascending;
                    DPRINT << "COMPUTE: stage = " << stage << ", i = " << i << ",  dir = " << (uint32_t)dir << ENDL();

                    if (j >= global_tile_start && j < global_tile_end) {
                        if (j > i) {
                            // Local sorting - both tiles in core memory

                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i - global_tile_start;
                            const uint32_t right_tile_id = j - global_tile_start;

                            tile_regs_acquire();

                            // Copy value tiles to DST register
                            COPY_TILE_TO_DST_INIT(input_tensor_transposed_cb_index, global_old_cb);
                            copy_tile(input_tensor_transposed_cb_index, left_tile_id, input_dest_start);
                            copy_tile(input_tensor_transposed_cb_index, right_tile_id, input_dest_end);

                            // Copy index tiles to DST register
                            COPY_TILE_TO_DST_INIT(index_tensor_transposed_cb_index, global_old_cb);
                            copy_tile(index_tensor_transposed_cb_index, left_tile_id, index_dest_start);
                            copy_tile(index_tensor_transposed_cb_index, right_tile_id, index_dest_end);

                            // constexpr uint32_t K = Wt * 16;
                            // constexpr uint32_t logK = ilog2(K);

                            uint32_t tile_input_low = input_dest_start;
                            uint32_t tile_input_high = input_dest_end;
                            uint32_t tile_index_low = index_dest_start;
                            uint32_t tile_index_high = index_dest_end;

                            if (sub == 1) {
                                ckernel::topk_local_sort(0, (int)dir, 5);
                            } else {
                                ckernel::topk_merge(0, m_iter, 32);

                                // uint32_t select_lower = dir;
                                if (dir) {
                                    tile_input_low = input_dest_end;
                                    tile_input_high = input_dest_start;
                                    tile_index_low = index_dest_end;
                                    tile_index_high = index_dest_start;
                                }
                            }
                            // ckernel::topk_rebuild(0, (uint32_t)dir, m_iter, K, logK, false);

                            //

                            // ckernel::topk_merge(0, m_iter, K);

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
                        DPRINT << "COMPUTE: All tiles" << ENDL();

                        const uint32_t tile_id = i - global_tile_start;  // TODO: Compute correct index
                        constexpr uint32_t FIRST_TILE = 0;

                        DPRINT << "COMPUTE: exchanging tiles " << i << " (tile id = " << tile_id << ") and " << j
                               << ENDL();

                        // Send tiles to other core
                        tile_regs_acquire();

                        // Copy index tiles to DST register for exchange
                        COPY_TILE_TO_DST_INIT(index_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(index_tensor_transposed_cb_index, tile_id, index_dest_start);

                        // Copy value tiles to DST register for exchange
                        COPY_TILE_TO_DST_INIT(input_tensor_transposed_cb_index, global_old_cb);
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
                        COPY_TILE_TO_DST_INIT(index_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(index_tensor_transposed_cb_index, tile_id, index_dest_start);

                        // Prepare local value tiles for sorting with new tiles
                        COPY_TILE_TO_DST_INIT(input_tensor_transposed_cb_index, global_old_cb);
                        copy_tile(input_tensor_transposed_cb_index, tile_id, input_dest_start);

                        cb_wait_front(index_tensor_peer_cb_index, one_tile);

                        // Load new index tile for sorting
                        COPY_TILE_TO_DST_INIT(index_tensor_peer_cb_index, global_old_cb);
                        copy_tile(index_tensor_peer_cb_index, FIRST_TILE, index_dest_end);

                        cb_pop_front(index_tensor_peer_cb_index, one_tile);

                        // Read other tile from writer
                        cb_wait_front(value_tensor_peer_cb_index, one_tile);

                        // Load new value tile for sorting
                        COPY_TILE_TO_DST_INIT(value_tensor_peer_cb_index, global_old_cb);
                        copy_tile(value_tensor_peer_cb_index, FIRST_TILE, input_dest_end);

                        cb_pop_front(value_tensor_peer_cb_index, one_tile);

                        // ckernel::topk_local_sort(0, (int)dir, 5);
                        ckernel::topk_merge(0, m_iter, 32);

                        uint32_t select_lower = dir ^ (i < j);

                        // TODO: Fix output tile selection w.r.t ascending
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

                        sync_packer_unpacker(packer_unpacker_sync_cb_index);
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
    DPRINT << "COMPUTE: Finished reading and sorting tiles." << ENDL();  // TODO: Remove
}  // MAIN
}  // namespace NAMESPACE
