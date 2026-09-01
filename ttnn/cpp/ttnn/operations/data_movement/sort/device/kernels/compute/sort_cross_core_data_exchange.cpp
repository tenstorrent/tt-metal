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
#include "api/compute/binary_max_min.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "sort_common.hpp"

#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t compute_with_storage_grid_size_y = get_arg(args::compute_with_storage_grid_size_y);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t number_of_tiles_per_core = get_arg(args::number_of_tiles_per_core);
    constexpr uint32_t number_of_cores_used = get_arg(args::number_of_cores_used);
    constexpr bool ascending = get_arg(args::ascending) == 1;

    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_tensor_dfb(dfb::index_tensor);
    DataflowBuffer input_tensor_transposed_dfb(dfb::input_tensor_transposed);
    DataflowBuffer index_tensor_transposed_dfb(dfb::index_tensor_transposed);
    DataflowBuffer value_tensor_intermediate_dfb(dfb::value_tensor_intermediate);
    DataflowBuffer index_tensor_intermediate_dfb(dfb::index_tensor_intermediate);
    DataflowBuffer value_tensor_peer_dfb(dfb::value_tensor_peer);
    DataflowBuffer index_tensor_peer_dfb(dfb::index_tensor_peer);
    DataflowBuffer packer_unpacker_sync_dfb(dfb::packer_unpacker_sync);
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_input_dfb(dfb::rm_input);
    DataflowBuffer rm_value_output_dfb(dfb::rm_value_output);
    DataflowBuffer rm_index_output_dfb(dfb::rm_index_output);
    DataflowBuffer rm_post_sort_index_dfb(dfb::rm_post_sort_index);
#else
    DataflowBuffer value_tensor_dfb(dfb::value_tensor);
    DataflowBuffer index_tensor_output_dfb(dfb::index_tensor_output);
#endif

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

    // LLK setup. The IS_ROW_MAJOR path re-inits for the transposed CB, so it issues a
    // second compute_kernel_hw_startup; each preserves the pre-cleanup binary_op_init_common
    // re-init (compute_kernel_hw_startup is documented call-once, but the pre-existing
    // mid-kernel re-init pattern is preserved as-is by the init-cleanup rename).
#ifdef IS_ROW_MAJOR
    compute_kernel_hw_startup(dfb::rm_input, dfb::index_tensor, dfb::input_tensor);
    // TODO(#52395): compute_kernel_hw_startup is a call-once API and should be the kernel's first Tensix-engine call, but here it follows another engine op (init_sfpu / a prior startup); see the issue.
    compute_kernel_hw_startup(dfb::input_tensor, dfb::index_tensor, dfb::input_tensor_transposed);
#else
    compute_kernel_hw_startup(dfb::input_tensor, dfb::index_tensor, dfb::input_tensor);
#endif
    ckernel::topk_tile_init();
    transpose_init(dfb::input_tensor);

    for (uint32_t h = 0; h < Ht; h++) {
#ifdef IS_ROW_MAJOR
        {
            constexpr uint32_t TILE_H = 32;
            tilize_init(dfb::rm_input, number_of_tiles_per_core, dfb::input_tensor);
            rm_input_dfb.wait_front(TILE_H);
            input_tensor_dfb.reserve_back(number_of_tiles_per_core);
            tilize_block(dfb::rm_input, number_of_tiles_per_core, dfb::input_tensor);
            input_tensor_dfb.push_back(number_of_tiles_per_core);
            rm_input_dfb.pop_front(TILE_H);
            tilize_uninit(dfb::rm_input, dfb::input_tensor);
        }
#endif

        bool dir = ascending ^ ((core_id & 1) == 1);

        // Read input value data
        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_dfb,
            index_tensor_dfb,
            input_tensor_transposed_dfb,
            index_tensor_transposed_dfb,
            number_of_tiles_per_core,
            /*switch_dir=*/true,
            dir,
            /*end_phase(log2(K))=*/5);

        global_old_cb = dfb::index_tensor;

        // Wait for bitonic sequence of Wt tiles
        input_tensor_transposed_dfb.wait_front(number_of_tiles_per_core);
        index_tensor_transposed_dfb.wait_front(number_of_tiles_per_core);

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

                    sync_packer_unpacker(packer_unpacker_sync_dfb);

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
                            copy_tile_to_dst_init_with_cb_update(dfb::input_tensor_transposed, global_old_cb);
                            copy_tile(dfb::input_tensor_transposed, left_tile_id, input_dest_start);
                            copy_tile(dfb::input_tensor_transposed, right_tile_id, input_dest_end);

                            // Copy index tiles to DST register
                            copy_tile_to_dst_init_with_cb_update(dfb::index_tensor_transposed, global_old_cb);
                            copy_tile(dfb::index_tensor_transposed, left_tile_id, index_dest_start);
                            copy_tile(dfb::index_tensor_transposed, right_tile_id, index_dest_end);

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
                            // UInt16-in-32b-DEST: mode-9 packer fixup before packing values (#50215).
                            prepare_uint16_fp32_dest_value_tiles_for_pack(tile_input_low, tile_input_high);
                            tile_regs_commit();
                            tile_regs_wait();

                            // Pack value tiles to the transposed buffer
                            pack_reconfig_data_format(dfb::input_tensor_transposed);
                            pack_tile<true>(tile_input_low, dfb::input_tensor_transposed, left_tile_id);
                            pack_tile<true>(tile_input_high, dfb::input_tensor_transposed, right_tile_id);

                            // Pack index tiles to the transposed buffer
                            pack_reconfig_data_format(dfb::index_tensor_transposed);
                            pack_tile<true>(tile_index_low, dfb::index_tensor_transposed, left_tile_id);
                            pack_tile<true>(tile_index_high, dfb::index_tensor_transposed, right_tile_id);

                            tile_regs_release();
                        }
                    } else {
                        const uint32_t tile_id = i - global_tile_start;
                        constexpr uint32_t FIRST_TILE = 0;

                        if ((i & 1) == 0) {  // i % 2
                            value_tensor_intermediate_dfb.reserve_back(one_tile);
                            index_tensor_intermediate_dfb.reserve_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb, index_tensor_transposed_dfb, tile_id, index_tensor_intermediate_dfb);
                            index_tensor_intermediate_dfb.push_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb,
                                input_tensor_transposed_dfb,
                                tile_id,
                                value_tensor_intermediate_dfb,
                                0,
                                /*prepare_uint16_value_for_pack=*/true);
                            value_tensor_intermediate_dfb.push_back(one_tile);

                            value_tensor_intermediate_dfb.reserve_back(one_tile);
                            index_tensor_intermediate_dfb.reserve_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb, index_tensor_transposed_dfb, tile_id + 1, index_tensor_intermediate_dfb);
                            index_tensor_intermediate_dfb.push_back(one_tile);

                            copy_tile_between_cbs(
                                global_old_cb,
                                input_tensor_transposed_dfb,
                                tile_id + 1,
                                value_tensor_intermediate_dfb,
                                0,
                                /*prepare_uint16_value_for_pack=*/true);
                            value_tensor_intermediate_dfb.push_back(one_tile);
                            sync_packer_unpacker(packer_unpacker_sync_dfb);
                        }

                        // Process received tiles from other core
                        //
                        // Both cores of the pair run this same merge on the same two tiles and each
                        // keeps one half of the result, so the two runs must agree on which half is
                        // which. topk_merge only swaps DEST[0]/DEST[1] when the values are strictly
                        // out of order, so for tied values the halves are told apart purely by which
                        // DEST slot each tile was loaded into. Loading "local" into DEST[0] would
                        // make that slot mean tile i on one core and tile j on the other: on a tie
                        // neither core swaps, the opposite `select_lower` values then select the
                        // same physical tile on both, and that tile's indices are duplicated while
                        // the partner's are lost (#54767).
                        //
                        // Ordering the slots by global tile id instead makes both cores build an
                        // identical DEST, so a tie leaves each core holding a different tile. For
                        // distinct values this is a no-op: a compare-exchange leaves min in DEST[0]
                        // and max in DEST[1] whichever slot each operand arrived in.
                        const bool local_tile_is_low = i < j;
                        const uint32_t local_value_dest = local_tile_is_low ? input_dest_start : input_dest_end;
                        const uint32_t local_index_dest = local_tile_is_low ? index_dest_start : index_dest_end;
                        const uint32_t peer_value_dest = local_tile_is_low ? input_dest_end : input_dest_start;
                        const uint32_t peer_index_dest = local_tile_is_low ? index_dest_end : index_dest_start;

                        tile_regs_acquire();

                        // Prepare local index tiles for sorting with new tiles
                        copy_tile_to_dst_init_with_cb_update(dfb::index_tensor_transposed, global_old_cb);
                        copy_tile(dfb::index_tensor_transposed, tile_id, local_index_dest);

                        // Prepare local value tiles for sorting with new tiles
                        copy_tile_to_dst_init_with_cb_update(dfb::input_tensor_transposed, global_old_cb);
                        copy_tile(dfb::input_tensor_transposed, tile_id, local_value_dest);

                        index_tensor_peer_dfb.wait_front(one_tile);

                        // Load new index tile for sorting
                        copy_tile_to_dst_init_with_cb_update(dfb::index_tensor_peer, global_old_cb);
                        copy_tile(dfb::index_tensor_peer, FIRST_TILE, peer_index_dest);

                        index_tensor_peer_dfb.pop_front(one_tile);

                        // Read other tile from writer
                        value_tensor_peer_dfb.wait_front(one_tile);

                        // Load new value tile for sorting
                        copy_tile_to_dst_init_with_cb_update(dfb::value_tensor_peer, global_old_cb);
                        copy_tile(dfb::value_tensor_peer, FIRST_TILE, peer_value_dest);

                        value_tensor_peer_dfb.pop_front(one_tile);

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

                        // UInt16-in-32b-DEST: mode-9 packer fixup before packing values (#50215).
                        prepare_uint16_fp32_dest_value_tile_for_pack(value_output_tile);

                        tile_regs_commit();
                        tile_regs_wait();

                        // Pack sorted index tiles to the transposed buffer
                        pack_reconfig_data_format(dfb::index_tensor_transposed);
                        pack_tile<true>(index_output_tile, dfb::index_tensor_transposed, tile_id);

                        // Pack sorted value tiles to the transposed buffer
                        pack_reconfig_data_format(dfb::input_tensor_transposed);
                        pack_tile<true>(value_output_tile, dfb::input_tensor_transposed, tile_id);

                        tile_regs_release();
                    }
                }  // Wt loop
            }  // sub loop
        }  // stages loop

        input_tensor_transposed_dfb.reserve_back(number_of_tiles_per_core);
        index_tensor_transposed_dfb.reserve_back(number_of_tiles_per_core);

        input_tensor_transposed_dfb.pop_front(number_of_tiles_per_core);
        index_tensor_transposed_dfb.pop_front(number_of_tiles_per_core);

        input_tensor_transposed_dfb.push_back(number_of_tiles_per_core);
        index_tensor_transposed_dfb.push_back(number_of_tiles_per_core);

#ifndef IS_ROW_MAJOR
        transpose_and_pack(
            input_tensor_transposed_dfb,
            value_tensor_dfb,
            number_of_tiles_per_core,
            /*prepare_uint16_value_for_pack=*/true);
        transpose_and_pack(index_tensor_transposed_dfb, index_tensor_output_dfb, number_of_tiles_per_core);
#else
        {
            // ROW_MAJOR output: un-transpose the sorted tiles back into the
            // PACK-only tile-format buffers (which are now empty after the
            // tilize_block/sort loops drained them), then pack_untilize them
            // into TILE_H RM rows for the writer/reader to drain.
            constexpr uint32_t TILE_H = 32;
            // DST_ACCUM_MODE is a compile-time macro injected by the framework when
            // enable_32_bit_dest=true is set on the compute hardware config (controlled by
            // is_32_bit_data in sort_program_factory.cpp: true for Float32 input or
            // UInt32 index). MAX_DEST_TILES is therefore data-format-dependent:
            // 32-bit DEST holds 4 tiles; 16-bit (BF16) DEST holds 8 tiles.
            constexpr uint32_t MAX_DEST_TILES = DST_ACCUM_MODE ? 4 : 8;
            // number_of_tiles_per_core is a power-of-two: get_number_of_tiles_per_core()
            // returns Wt / num_cores, and Wt is always a power-of-two (padded by
            // pre_sort_transform_tensor). MAX_DEST_TILES is also a power-of-two (4 or 8),
            // so number_of_tiles_per_core % SUB_BLOCK_DIM == 0 is always satisfied.
            constexpr uint32_t SUB_BLOCK_DIM =
                (number_of_tiles_per_core < MAX_DEST_TILES) ? number_of_tiles_per_core : MAX_DEST_TILES;
            constexpr uint32_t NUM_SUB_BLOCKS = number_of_tiles_per_core / SUB_BLOCK_DIM;
            static_assert(
                number_of_tiles_per_core % SUB_BLOCK_DIM == 0,
                "number_of_tiles_per_core must be divisible by SUB_BLOCK_DIM");

            transpose_and_pack(
                input_tensor_transposed_dfb,
                input_tensor_dfb,
                number_of_tiles_per_core,
                /*prepare_uint16_value_for_pack=*/true);

            transpose_and_pack(index_tensor_transposed_dfb, rm_post_sort_index_dfb, number_of_tiles_per_core);

            // Untilize values: number_of_tiles_per_core tiles → TILE_H RM pages.
            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
            compute_kernel_hw_startup(dfb::input_tensor, dfb::index_tensor, dfb::rm_value_output);
            pack_untilize_init<SUB_BLOCK_DIM, number_of_tiles_per_core>(dfb::input_tensor, dfb::rm_value_output);
            input_tensor_dfb.wait_front(number_of_tiles_per_core);
            rm_value_output_dfb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, number_of_tiles_per_core>(
                    dfb::input_tensor, 1, dfb::rm_value_output, b);
                input_tensor_dfb.pop_front(SUB_BLOCK_DIM);
            }
            rm_value_output_dfb.push_back(TILE_H);
            pack_untilize_uninit(dfb::rm_value_output);

            // Untilize indices: number_of_tiles_per_core tiles → TILE_H RM pages.
            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
            compute_kernel_hw_startup(dfb::rm_post_sort_index, dfb::input_tensor, dfb::rm_index_output);
            pack_untilize_init<SUB_BLOCK_DIM, number_of_tiles_per_core>(dfb::rm_post_sort_index, dfb::rm_index_output);
            rm_post_sort_index_dfb.wait_front(number_of_tiles_per_core);
            rm_index_output_dfb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, number_of_tiles_per_core>(
                    dfb::rm_post_sort_index, 1, dfb::rm_index_output, b);
                rm_post_sort_index_dfb.pop_front(SUB_BLOCK_DIM);
            }
            rm_index_output_dfb.push_back(TILE_H);
            pack_untilize_uninit(dfb::rm_index_output);
        }
#endif
    }  // h loop
}  // void kernel_main()
