// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "sort_common.hpp"

void kernel_main() {
    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t number_of_available_cores = get_arg(args::number_of_available_cores);
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr bool descending = get_arg(args::descending);
    constexpr bool stable =
        get_arg(args::stable);  // TODO: In the future change LLK to have the option or add additional step with
                                // checking values and indexes after the sorting
                                // Issue: https://github.com/tenstorrent/tt-metal/issues/20625
    constexpr uint32_t log2Wt = get_arg(args::log2Wt);

    // The ROW_MAJOR buffers exist only in the ROW_MAJOR configuration, so the layout gate arrives as
    // a preprocessor define. This mirror keeps the ordinary runtime conditions below readable.
#ifdef IS_ROW_MAJOR
    constexpr bool is_row_major = true;
#else
    constexpr bool is_row_major = false;
#endif

    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_tensor_dfb(dfb::index_tensor);
    DataflowBuffer input_tensor_transposed_dfb(dfb::input_tensor_transposed);
    DataflowBuffer index_tensor_transposed_dfb(dfb::index_tensor_transposed);
    DataflowBuffer input_tensor_output_dfb(dfb::input_tensor_output);
    DataflowBuffer index_tensor_output_dfb(dfb::index_tensor_output);
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_input_value_dfb(dfb::rm_input_value);
    DataflowBuffer rm_input_index_dfb(dfb::rm_input_index);
    DataflowBuffer rm_output_value_dfb(dfb::rm_output_value);
    DataflowBuffer rm_output_index_dfb(dfb::rm_output_index);
#endif

    // Constants
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    constexpr uint32_t TILE_H = 32;

    // For ROW_MAJOR, compute_kernel_hw_startup initialises the MATH-PACK DST
    // semaphore required by tilize_block before the first pair is processed.
    // For TILE layout the existing topk_tile_init + transpose_init path is used.
#ifdef IS_ROW_MAJOR
    compute_kernel_hw_startup(dfb::rm_input_value, dfb::rm_input_index, dfb::input_tensor);
#else
    compute_kernel_hw_startup(dfb::input_tensor, dfb::input_tensor_transposed);
    ckernel::topk_tile_init();
    transpose_init(dfb::input_tensor);
#endif

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
#ifdef IS_ROW_MAJOR
                            // rm_input_value and rm_input_index hold TILE_H pair-rows: each entry is
                            // 2 * TILE_W elements wide (left tile row concatenated with right tile
                            // row). tilize_block(buffer, 2, out) reads 2048 elements as 32 rows of
                            // 64 elements, producing 2 tiles.
                            tilize_init(dfb::rm_input_value, 2, dfb::input_tensor);
                            rm_input_value_dfb.wait_front(TILE_H);
                            input_tensor_dfb.reserve_back(2);
                            tilize_block(dfb::rm_input_value, 2, dfb::input_tensor);
                            input_tensor_dfb.push_back(2);
                            rm_input_value_dfb.pop_front(TILE_H);
                            tilize_uninit(dfb::rm_input_value, dfb::input_tensor);
                            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init
                            // (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
                            compute_kernel_hw_startup(dfb::rm_input_index, dfb::rm_input_index, dfb::index_tensor);

                            tilize_init(dfb::rm_input_index, 2, dfb::index_tensor);
                            rm_input_index_dfb.wait_front(TILE_H);
                            index_tensor_dfb.reserve_back(2);
                            tilize_block(dfb::rm_input_index, 2, dfb::index_tensor);
                            index_tensor_dfb.push_back(2);
                            rm_input_index_dfb.pop_front(TILE_H);
                            tilize_uninit(dfb::rm_input_index, dfb::index_tensor);
                            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init
                            // (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
                            compute_kernel_hw_startup(
                                dfb::input_tensor, dfb::index_tensor, dfb::input_tensor_transposed);

                            ckernel::topk_tile_init();
                            transpose_init(dfb::input_tensor);
#endif

                            input_tensor_dfb.wait_front(2 * one_tile);
                            index_tensor_dfb.wait_front(2 * one_tile);

                            tile_regs_acquire();
                            // For RM, tiles from tilize are always regular (non-transposed),
                            // so we always transpose before sort (same as the stage==1 sub==1
                            // branch in the TILE path). For TILE, tiles are pre-transposed
                            // in all stages except the very first.
                            if ((stage == 1 && sub == 1) || is_row_major) {
                                reconfig_data_format_srca(dfb::input_tensor);
                                transpose_init(dfb::input_tensor);
                                transpose_tile(dfb::input_tensor, 0, input_dest_start);
                                transpose_tile(dfb::input_tensor, 1, input_dest_end);

                                // Process index tiles
                                reconfig_data_format_srca(dfb::index_tensor);
                                transpose_init(dfb::index_tensor);
                                transpose_tile(dfb::index_tensor, 0, index_dest_start);
                                transpose_tile(dfb::index_tensor, 1, index_dest_end);
                            } else {
                                // Intermediate step - tiles are already transposed
                                // Process value tiles
                                reconfig_data_format_srca(dfb::input_tensor);
                                copy_init(dfb::input_tensor);
                                copy_tile(dfb::input_tensor, 0, input_dest_start);
                                copy_tile(dfb::input_tensor, 1, input_dest_end);

                                // Process index tiles
                                reconfig_data_format_srca(dfb::index_tensor);
                                copy_init(dfb::index_tensor);
                                copy_tile(dfb::index_tensor, 0, index_dest_start);
                                copy_tile(dfb::index_tensor, 1, index_dest_end);
                            }

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

                            // UInt16-in-32b-DEST: mode-9 packer fixup before packing values (#50215).
                            prepare_uint16_fp32_dest_value_tiles_for_pack(tile_input_low, tile_input_high);

                            tile_regs_commit();

                            input_tensor_dfb.pop_front(2 * one_tile);
                            index_tensor_dfb.pop_front(2 * one_tile);

                            // For RM: always transpose back so the packed tile is in
                            // regular (non-transposed) format, which pack_untilize_block
                            // expects. For TILE: only do the transpose-back at the last
                            // stage/sub (intermediate stages keep the transposed format in
                            // DRAM to avoid an extra transpose next stage).
                            if ((stage == log2Wt && sub == 1) || is_row_major) {
                                input_tensor_transposed_dfb.reserve_back(2 * one_tile);
                                index_tensor_transposed_dfb.reserve_back(2 * one_tile);

                                tile_regs_wait();
                                pack_reconfig_data_format(dfb::input_tensor_transposed);
                                pack_tile(tile_input_low, dfb::input_tensor_transposed);
                                pack_tile(tile_input_high, dfb::input_tensor_transposed);

                                pack_reconfig_data_format(dfb::index_tensor_transposed);
                                pack_tile(tile_index_low, dfb::index_tensor_transposed);
                                pack_tile(tile_index_high, dfb::index_tensor_transposed);
                                tile_regs_release();

                                input_tensor_transposed_dfb.push_back(2 * one_tile);
                                index_tensor_transposed_dfb.push_back(2 * one_tile);

                                // Pack and push sorted values tensor tiles
                                input_tensor_transposed_dfb.wait_front(2 * one_tile);

                                tile_regs_acquire();
                                reconfig_data_format_srca(dfb::input_tensor_transposed);
                                transpose_init(dfb::input_tensor_transposed);
                                transpose_tile(dfb::input_tensor_transposed, 0, input_dest_start);
                                transpose_tile(dfb::input_tensor_transposed, 1, input_dest_end);
                                // UInt16-in-32b-DEST: transpose re-unpacks into low 16; fix up before pack (#50215).
                                prepare_uint16_fp32_dest_value_tiles_for_pack(input_dest_start, input_dest_end);
                                tile_regs_commit();

                                input_tensor_transposed_dfb.pop_front(2 * one_tile);

                                input_tensor_output_dfb.reserve_back(2 * one_tile);

                                tile_regs_wait();
                                pack_reconfig_data_format(dfb::input_tensor_output);
                                pack_tile(input_dest_start, dfb::input_tensor_output);
                                pack_tile(input_dest_end, dfb::input_tensor_output);
                                tile_regs_release();

                                // Push tiles to writer
                                input_tensor_output_dfb.push_back(2 * one_tile);

                                // Pack and push adjusted index tensor tiles
                                index_tensor_transposed_dfb.wait_front(2 * one_tile);

                                tile_regs_acquire();
                                reconfig_data_format_srca(dfb::index_tensor_transposed);
                                transpose_init(dfb::index_tensor_transposed);
                                transpose_tile(dfb::index_tensor_transposed, 0, input_dest_start);
                                transpose_tile(dfb::index_tensor_transposed, 1, input_dest_end);
                                tile_regs_commit();

                                index_tensor_transposed_dfb.pop_front(2 * one_tile);

                                index_tensor_output_dfb.reserve_back(2 * one_tile);

                                tile_regs_wait();
                                pack_reconfig_data_format(dfb::index_tensor_output);
                                pack_tile(input_dest_start, dfb::index_tensor_output);
                                pack_tile(input_dest_end, dfb::index_tensor_output);
                                tile_regs_release();

                                index_tensor_output_dfb.push_back(2 * one_tile);
                            } else {
                                // Intermediate step - pack and push transposed tiles to be saved for the next stage
                                index_tensor_output_dfb.reserve_back(2 * one_tile);
                                input_tensor_output_dfb.reserve_back(2 * one_tile);

                                tile_regs_wait();
                                // Process value tiles
                                pack_reconfig_data_format(dfb::input_tensor_output);
                                pack_tile(tile_input_low, dfb::input_tensor_output);
                                pack_tile(tile_input_high, dfb::input_tensor_output);

                                pack_reconfig_data_format(dfb::index_tensor_output);
                                pack_tile(tile_index_low, dfb::index_tensor_output);
                                pack_tile(tile_index_high, dfb::index_tensor_output);
                                tile_regs_release();

                                input_tensor_output_dfb.push_back(2 * one_tile);
                                index_tensor_output_dfb.push_back(2 * one_tile);
                            }

#ifdef IS_ROW_MAJOR
                            // pack_untilize_init already calls llk_pack_reconfig_data_format
                            // internally, so a pack_reconfig_data_format call here would be a
                            // no-op. What is actually needed, and what pack_untilize_init does not
                            // provide, is a full-word PCK_DEST_RD_CTRL reset (llk_pack_hw_configure)
                            // and a W-counter / dest-offset reset (llk_pack_dest_init) to clear the
                            // state left by the preceding index pack_tile. There is no public API
                            // for just those two things yet, so compute_kernel_hw_startup is used as
                            // a workaround.
                            compute_kernel_hw_startup(dfb::input_tensor, dfb::index_tensor, dfb::rm_output_value);
                            // Untilize the 2 sorted tiles into TILE_H pair-rows, matching the
                            // reader's pair-row input layout. block_ct_dim=2, block_rt_dim=1
                            // produces TILE_H rows of 2*TILE_W elements each (left tile columns
                            // followed by right tile columns within each row). The writer splits
                            // each entry back into two DRAM half-row writes.
                            pack_untilize_init<2>(dfb::input_tensor_output, dfb::rm_output_value);
                            input_tensor_output_dfb.wait_front(2);
                            rm_output_value_dfb.reserve_back(TILE_H);
                            pack_untilize_block<2>(dfb::input_tensor_output, 1, dfb::rm_output_value);
                            input_tensor_output_dfb.pop_front(2);
                            rm_output_value_dfb.push_back(TILE_H);
                            pack_untilize_uninit(dfb::rm_output_value);
                            // Reconfig the packer to the index-output format before the index
                            // untilize. The output operand must be the untilize destination
                            // (rm_output_index), not its source (index_tensor_output): those two
                            // happen to share a data format today, so passing the source only works
                            // by coincidence.
                            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init
                            // (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
                            compute_kernel_hw_startup(
                                dfb::rm_input_index, dfb::rm_input_index, dfb::rm_output_index);

                            pack_untilize_init<2>(dfb::index_tensor_output, dfb::rm_output_index);
                            index_tensor_output_dfb.wait_front(2);
                            rm_output_index_dfb.reserve_back(TILE_H);
                            pack_untilize_block<2>(dfb::index_tensor_output, 1, dfb::rm_output_index);
                            index_tensor_output_dfb.pop_front(2);
                            rm_output_index_dfb.push_back(TILE_H);
                            pack_untilize_uninit(dfb::rm_output_index);
                            // Reset compute state for the next pair's tilize.
                            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init
                            // (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
                            compute_kernel_hw_startup(dfb::rm_input_value, dfb::rm_input_index, dfb::input_tensor);
#endif

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
