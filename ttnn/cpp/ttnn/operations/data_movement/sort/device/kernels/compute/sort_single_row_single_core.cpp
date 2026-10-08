// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
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

#ifdef SORT_STABLE_FUSED_32B_DEST
#include "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_fused_raw16.hpp"

// The stable path's row in fused [bf16 value | u16 index] keys, in the 32-bit DEST and a 32-bit buffer from its first
// local sort to its last merge: the plain network gives the comparator's order. The fuse makes every zero and denormal
// +0, and a NaN becomes the infinity of its sign after the first local sort, where this kernel's comparator path does
// both.
FORCE_INLINE void enter_fused_section() { ckernel::topk_tile_init</*fused=*/true>(); }

FORCE_INLINE void leave_fused_section() { ckernel::topk_tile_init(); }

// transpose_and_pack for u16 tiles in the 32-bit DEST: the datums move to the packer's half before the pack.
FORCE_INLINE void transpose_and_pack_u16(DataflowBuffer& transposed_dfb, DataflowBuffer& dest_dfb, const uint32_t Wt) {
    constexpr uint32_t one_tile = 1;
    reconfig_data_format_srca(transposed_dfb.get_id());
    transpose_init(transposed_dfb.get_id());
    pack_reconfig_data_format(dest_dfb.get_id());
    transposed_dfb.wait_front(Wt);
    for (uint32_t i = 0; i < Wt; ++i) {
        tile_regs_acquire();
        dest_dfb.reserve_back(one_tile);
        transpose_tile(transposed_dfb.get_id(), i, 0);
        topk_uint16_move_dest_tile_to_pack_half(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dest_dfb.get_id());
        dest_dfb.push_back(one_tile);
        tile_regs_release();
    }
    transposed_dfb.pop_front(Wt);
}

// The row's tile pairs transposed, fused and sorted in alternating directions into the key buffer.
template <bool largest>
void sort_row_pairs_to_keys(
    DataflowBuffer& input_dfb,
    DataflowBuffer& index_dfb,
    DataflowBuffer& keys_dfb,
    const uint32_t Wt,
    const bool ascending) {
    reconfig_data_format_srca(dfb::index_tensor);
    pack_reconfig_data_format(dfb::fused_keys);
    keys_dfb.reserve_back(Wt);
    bool ascending_local = ascending;
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        tile_regs_acquire();

        input_dfb.wait_front(2);
        index_dfb.wait_front(2);

        transpose_init<true>(dfb::index_tensor);
        transpose_tile<true>(dfb::input_tensor, 0, 0);
        transpose_tile<true>(dfb::input_tensor, 1, 1);
        transpose_tile<true>(dfb::index_tensor, 0, 2);
        transpose_tile<true>(dfb::index_tensor, 1, 3);
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            topk_fused_raw16::fuse_raw16_slab<largest, topk_fused_raw16::Zeros::ToPositive>,
            0,
            VectorMode::RC_custom)));
        ckernel::topk_local_sort</*stable_sort=*/false, /*is_fp32_dest_acc_en=*/true, /*fused=*/true>(
            0, (int)ascending_local, /*i_end_phase=*/5);
        MATH((_llk_math_eltwise_unary_sfpu_params_(topk_fused_raw16::keys_nan_to_inf, 0, VectorMode::RC_custom)));

        tile_regs_commit<true>();
        tile_regs_wait();

        pack_tile<true, true>(0, dfb::fused_keys, wt);
        pack_tile<true, true>(1, dfb::fused_keys, wt + 1);
        input_dfb.pop_front(2);
        index_dfb.pop_front(2);

        tile_regs_release<true>();

        ascending_local = !ascending_local;
    }
    keys_dfb.push_back(Wt);
}

// One pass of a merge stage on the keys in place: the plain merge network at distance 2^(sub - 1) tiles, or for sub 1
// the local sort of each tile pair.
void merge_pass_keys(
    DataflowBuffer& synchronization_dfb,
    const uint32_t Wt,
    const uint32_t stage,
    const uint32_t sub,
    const bool ascending) {
    constexpr uint32_t one_tile = 1;
    const uint32_t sub_dist = 1 << (sub - 1);
    for (uint32_t i = 0; i < Wt; i++) {
        const uint32_t j = i ^ sub_dist;
        if (j <= i) {
            continue;
        }
        const bool dir = (((i >> stage) & 1) == 0) == ascending;

        tile_regs_acquire();

        synchronization_dfb.wait_front(one_tile);
        synchronization_dfb.pop_front(one_tile);
        synchronization_dfb.reserve_back(one_tile);

        copy_init<true>(dfb::fused_keys);
        copy_tile<true>(dfb::fused_keys, i, 0);
        copy_tile<true>(dfb::fused_keys, j, 1);

        uint32_t low = 0;
        uint32_t high = 1;
        if (sub == 1) {
            ckernel::topk_local_sort</*stable_sort=*/false, /*is_fp32_dest_acc_en=*/true, /*fused=*/true>(
                0, (int)dir, /*i_end_phase=*/5);
        } else {
            ckernel::topk_merge</*idir=*/false, /*stable_sort=*/false, /*is_fp32_dest_acc_en=*/true, /*fused=*/true>(
                0, stage - 1, /*k=*/64);
            if (dir) {
                // The merge leaves the smaller keys in DEST 0 and the larger in DEST 1.
                low = 1;
                high = 0;
            }
        }

        tile_regs_commit<true>();
        tile_regs_wait();

        pack_tile<true, true>(low, dfb::fused_keys, i);
        pack_tile<true, true>(high, dfb::fused_keys, j);

        synchronization_dfb.push_back(one_tile);

        tile_regs_release<true>();
    }
}

// The row's keys split back into the transposed value and index buffers.
template <bool largest>
void split_keys(
    DataflowBuffer& keys_dfb,
    DataflowBuffer& input_transposed_dfb,
    DataflowBuffer& index_transposed_dfb,
    const uint32_t Wt) {
    pack_reconfig_data_format(dfb::index_tensor_transposed);
    input_transposed_dfb.reserve_back(Wt);
    index_transposed_dfb.reserve_back(Wt);
    for (uint32_t i = 0; i < Wt; i += 2) {
        tile_regs_acquire();

        copy_init<true>(dfb::fused_keys);
        copy_tile<true>(dfb::fused_keys, i, 0);
        copy_tile<true>(dfb::fused_keys, i + 1, 1);
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            topk_fused_raw16::defuse_raw16<largest, false>, 0, VectorMode::RC_custom, 2)));

        tile_regs_commit<true>();
        tile_regs_wait();

        pack_tile<true, true>(0, dfb::input_tensor_transposed, i);
        pack_tile<true, true>(1, dfb::input_tensor_transposed, i + 1);
        pack_tile<true, true>(2, dfb::index_tensor_transposed, i);
        pack_tile<true, true>(3, dfb::index_tensor_transposed, i + 1);

        tile_regs_release<true>();
    }
    keys_dfb.pop_front(Wt);
    input_transposed_dfb.push_back(Wt);
    index_transposed_dfb.push_back(Wt);
}
#endif

/*
This sorting algorithm is based on Bitonic Merge Sort and operates on input data arranged in tiles.

The algorithm processes the data such that the dimension to be sorted becomes the last dimension of the tensor.
From the perspective of tile arrangement, sorting is performed row by row in a matrix-like structure.

### Overview:
1. **Tile Initialization**:
    - A full row of tiles (size `Wt`) is read from DRAM into L1 memory.
    - Corresponding tiles containing the initial data indices are also generated.

2. **Sorting Mechanism**:
    - The core of the sorting is performed using `ckernel::topk_local_sort`, which:
      - Sorts two input tiles in-place.
      - Updates the indices of the data to reflect the new order.
    - Since `ckernel::topk_local_sort` operates on columns, an additional transposition step is required.
    - The number of tiles in the `Wt` dimension must be a multiple of 64 (2 * Tile_Width (32)) to ensure compatibility.

3. **Bitonic Sequence Formation**:
    - The function `sort_Wt_tiles_row_to_bitonic_sequence`:
      - Sorts pairs of tiles alternately in ascending and descending order.
      - Produces a set of sorted tile pairs with alternating sorting directions.

4. **Bitonic Merge Sort**:
    - The tiles are further sorted in stages to ensure the entire row is sorted.
    - At each stage, tile indices are calculated, and tiles are sorted pairwise.
    - This process continues until all tiles in the row are sorted.

5. **Multicore Calculation**:
    - Multicore parallelism is enabled by assigning each row of tiles (`Wt`) to a separate core.
    - If the number of rows (`Ht`) exceeds the number of available cores, the workload is distributed such that some
cores process multiple rows.
    - This ensures efficient utilization of all cores and minimizes idle time during computation.

6. **Final Steps**:
    - Once sorted, the tiles are transposed back to the desired dimension.
    - The sorted data is then written back to DRAM.

### Example:
- Input: A 64x128 matrix, represented as 2x4 tiles: T0, T1, T2, T3
                                                    T4, T5, T6, T7
- Sorting (ascending order):
0. Distributing workload across cores:
   - Core 0 processes T0, T1, T2, T3
   - Core 1 processes T4, T5, T6, T7
Calculation of each row:
  1. **Pairwise Sorting**:
      - T0 and T1 are sorted as a pair in ascending order.
      - T2 and T3 are sorted as a pair in descending order.
  2. **Sorting Across Pairs**:
      - **Stage 1**: T0 and T2 are sorted in ascending order, and T1 and T3 are sorted in ascending order.
      - **Stage 2**: T0 and T1 are sorted in ascending order, and T2 and T3 are sorted in ascending order.
  3. **Data Saving**:
      - The tiles are now fully sorted along the desired dimension and ready to be saved.
 */
void kernel_main() {
    // Runtime args
    const uint32_t core_loop_count = get_arg(args::core_loop_count);

    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr bool descending = get_arg(args::descending);
    // Comparator-stable network (issue #33492): on exact value ties the index tiles are
    // compare-exchanged so equal values keep their original (ascending-index) order, matching
    // torch.sort(stable=True) in both directions.
    constexpr bool stable = get_arg(args::stable);

    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_tensor_dfb(dfb::index_tensor);
    DataflowBuffer input_tensor_transposed_dfb(dfb::input_tensor_transposed);
    DataflowBuffer index_tensor_transposed_dfb(dfb::index_tensor_transposed);
    DataflowBuffer synchronization_dfb(dfb::synchronization);
#ifdef SORT_STABLE_FUSED_32B_DEST
    DataflowBuffer fused_keys_dfb(dfb::fused_keys);
#endif
#ifndef IS_ROW_MAJOR
    DataflowBuffer value_tensor_dfb(dfb::value_tensor);
    DataflowBuffer index_tensor_output_dfb(dfb::index_tensor_output);
#else
    DataflowBuffer rm_input_dfb(dfb::rm_input);
    DataflowBuffer rm_value_output_dfb(dfb::rm_value_output);
    DataflowBuffer rm_index_output_dfb(dfb::rm_index_output);
    // PACK-only DFB that holds the un-transposed sorted index tiles between the
    // sort phase and the index pack_untilize_block. Using a separate buffer here
    // (rather than reusing index_tensor, which is BRISC-pushed by the
    // writer) avoids a mixed-producer counter race: BRISC uses += into the L1
    // receive counter, while PACK overwrites it with PACK's own local counter,
    // so PACK's pushes silently clobber BRISC's pushes and wait_front spins
    // forever.
    DataflowBuffer rm_post_sort_index_dfb(dfb::rm_post_sort_index);
#endif

    constexpr uint32_t one_tile = 1;

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    // For TILE path: one-time initialisation before the loop.
    // For ROW_MAJOR: tilize/untilize are interleaved with sort each iteration.
    //
    // ROW_MAJOR requires compute_kernel_hw_startup to be called once before any
    // other compute API. This initialises the MATH-PACK DST semaphore (via
    // llk_math_pack_sync_init + llk_pack_dest_init) so that tilize_block's
    // internal llk_math_wait_for_dest_available() does not spin forever.
    // Without this call the kernel deadlocks on the first tilize_block invocation.
#ifdef IS_ROW_MAJOR
    compute_kernel_hw_startup(dfb::rm_input, dfb::index_tensor, dfb::input_tensor);
#else
    compute_kernel_hw_startup(dfb::input_tensor, dfb::input_tensor_transposed);
    ckernel::topk_tile_init();
    transpose_init(dfb::input_tensor);
#endif

    constexpr auto tie_order = ckernel::topk_tie_order_from_global_direction(descending);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const bool ascending = !descending;

        // ------------------------------------------------------------------
        // ROW_MAJOR: tilize one tile-row of RM data → Wt TILE-format tiles,
        // then reinitialise for sort.
        //
        // tilize_block uses out-of-order pack (llk_pack<true>) which writes
        // tiles to the buffer WITHOUT signalling them internally. The
        // explicit reserve_back / push_back pair is therefore required:
        //   • reserve_back: guarantees write slots are free before the
        //     out-of-order pack writes to them.
        //   • push_back:    signals the tiles as ready so that the
        //     subsequent sort_Wt_tiles_row_to_bitonic_sequence (which calls
        //     wait_front) does not deadlock.
        // ------------------------------------------------------------------
#ifdef IS_ROW_MAJOR
        {
            constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT (tt::constants not available in device kernels)
            tilize_init(dfb::rm_input, Wt, dfb::input_tensor);
            rm_input_dfb.wait_front(TILE_H);
            input_tensor_dfb.reserve_back(Wt);
            tilize_block(dfb::rm_input, Wt, dfb::input_tensor);
            input_tensor_dfb.push_back(Wt);
            rm_input_dfb.pop_front(TILE_H);
            tilize_uninit(dfb::rm_input, dfb::input_tensor);

            // Re-initialise compute hardware for the sort phase.
            //
            // tilize_uninit does not reset the PACK side on WormholeB0 (the
            // Blackhole-only llk_pack_init path is skipped), so the packer is
            // still configured for the out-of-order tilize writes it just
            // performed. This compute_kernel_hw_startup re-arms the MATH-PACK DST
            // semaphore (llk_math_pack_sync_init + llk_pack_dest_init) and resets
            // PACK to normal mode so that pack_tile / pack_reconfig_data_format
            // inside sort_Wt_tiles_row_to_bitonic_sequence work correctly. This is
            // a deliberate mid-kernel re-init that preserves the pre-cleanup
            // binary_op_init_common behaviour (same pattern as
            // layernorm_large_tensor.cpp's TILIZE_IN path). NOTE: compute_kernel_hw_startup
            // is documented call-once; correcting this re-init pattern is out of scope
            // for the init-cleanup rename and left to the sort kernel owners.
            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
            compute_kernel_hw_startup(dfb::input_tensor, dfb::index_tensor, dfb::input_tensor_transposed);
            ckernel::topk_tile_init();
            transpose_init(dfb::input_tensor);
        }
#endif

#ifdef SORT_STABLE_FUSED_32B_DEST
        enter_fused_section();
        sort_row_pairs_to_keys<descending>(input_tensor_dfb, index_tensor_dfb, fused_keys_dfb, Wt, ascending);
        fused_keys_dfb.wait_front(Wt);

        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }
        synchronization_dfb.reserve_back(one_tile);
        synchronization_dfb.push_back(one_tile);
        reconfig_data_format_srca(dfb::fused_keys);
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                merge_pass_keys(synchronization_dfb, Wt, stage, sub, ascending);
            }
        }
        synchronization_dfb.wait_front(one_tile);
        synchronization_dfb.pop_front(one_tile);

        split_keys<descending>(fused_keys_dfb, input_tensor_transposed_dfb, index_tensor_transposed_dfb, Wt);
        leave_fused_section();
#else
        sort_Wt_tiles_row_to_bitonic_sequence<stable, tie_order>(
            input_tensor_dfb,
            index_tensor_dfb,
            input_tensor_transposed_dfb,
            index_tensor_transposed_dfb,
            Wt,
            /*switch_dir=*/true,
            ascending,
            /*end_phase(log2(K))=*/5);

        // Wait for bitonic sequence of Wt tiles
        input_tensor_transposed_dfb.wait_front(Wt);
        index_tensor_transposed_dfb.wait_front(Wt);

        // Sort and merge step of bitonic merge sort
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }

        synchronization_dfb.reserve_back(one_tile);
        synchronization_dfb.push_back(one_tile);

        for (uint32_t stage = 2; stage <= stages; stage++) {
            const uint32_t m_iter = stage - 1;
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        // Determine direction for this comparison block
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir = ascending_block == ascending;

                        // Get indexes of tiles to compare
                        const uint32_t left_tile_id = i;
                        const uint32_t right_tile_id = j;

                        tile_regs_acquire();

                        synchronization_dfb.wait_front(one_tile);
                        synchronization_dfb.pop_front(one_tile);
                        synchronization_dfb.reserve_back(one_tile);

                        reconfig_data_format_srca(dfb::input_tensor_transposed, dfb::index_tensor_transposed);
                        copy_init(dfb::index_tensor_transposed);
                        copy_tile(dfb::index_tensor_transposed, left_tile_id, index_dest_start);
                        copy_tile(dfb::index_tensor_transposed, right_tile_id, index_dest_end);

                        reconfig_data_format_srca(dfb::index_tensor_transposed, dfb::input_tensor_transposed);
                        copy_init(dfb::input_tensor_transposed);
                        copy_tile(dfb::input_tensor_transposed, left_tile_id, input_dest_start);
                        copy_tile(dfb::input_tensor_transposed, right_tile_id, input_dest_end);

                        uint32_t tile_input_low = input_dest_start;
                        uint32_t tile_input_high = input_dest_end;
                        uint32_t tile_index_low = index_dest_start;
                        uint32_t tile_index_high = index_dest_end;

                        if (sub == 1) {
                            // Use sort LLK only the last stage to sort the last pair of tiles - speed up
                            ckernel::topk_local_sort<
                                stable,
                                DST_ACCUM_MODE,
                                /*fused=*/false,
                                /*rank_stamped=*/false,
                                tie_order>(
                                /*idst=*/0, (int)dir, /*end_phase(log2(K))=*/5);
                        } else {
                            ckernel::topk_merge<
                                /*idir=*/false,
                                stable,
                                DST_ACCUM_MODE,
                                /*fused=*/false,
                                /*rank_stamped=*/false,
                                tie_order>(/*idst=*/0, m_iter, /*k=*/64);

                            if (dir) {
                                // topk_merge puts smallest values in DEST[0] and largest in DEST[1]
                                // We swap their indices when using descending order
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

                        pack_reconfig_data_format(dfb::input_tensor_transposed);
                        pack_tile<true>(tile_input_low, dfb::input_tensor_transposed, left_tile_id);
                        pack_tile<true>(tile_input_high, dfb::input_tensor_transposed, right_tile_id);

                        pack_reconfig_data_format(dfb::index_tensor_transposed);
                        pack_tile<true>(tile_index_low, dfb::index_tensor_transposed, left_tile_id);
                        pack_tile<true>(tile_index_high, dfb::index_tensor_transposed, right_tile_id);

                        synchronization_dfb.push_back(one_tile);

                        tile_regs_release();
                    }
                }
            }
        }

        synchronization_dfb.wait_front(one_tile);
        synchronization_dfb.pop_front(one_tile);

        input_tensor_transposed_dfb.reserve_back(Wt);
        index_tensor_transposed_dfb.reserve_back(Wt);

        input_tensor_transposed_dfb.pop_front(Wt);
        index_tensor_transposed_dfb.pop_front(Wt);

        input_tensor_transposed_dfb.push_back(Wt);
        index_tensor_transposed_dfb.push_back(Wt);

#endif

        // TILE path: transpose-and-pack to 2-tile streaming buffers so the reader
        // and writer can stream tiles one-by-one to DRAM (existing behaviour).
        //
        // ROW_MAJOR path: transpose-and-pack to the now-empty Wt-tile buffers
        // (input_tensor for values, rm_post_sort_index for indices), then untilize
        // the full row back to RM pages.
#ifndef IS_ROW_MAJOR
        // Values tensor → 2-tile streaming buffer (writer drains to DRAM)
        transpose_and_pack(
            input_tensor_transposed_dfb, value_tensor_dfb, Wt, /*prepare_uint16_value_for_pack=*/true);
        // Index tensor → 2-tile streaming buffer (reader drains to DRAM)
#ifdef SORT_STABLE_FUSED_32B_DEST
        transpose_and_pack_u16(index_tensor_transposed_dfb, index_tensor_output_dfb, Wt);
#else
        transpose_and_pack(index_tensor_transposed_dfb, index_tensor_output_dfb, Wt);
#endif
#else
        {
            constexpr uint32_t TILE_H = 32;

            constexpr uint32_t MAX_DEST_TILES = DST_ACCUM_MODE ? 4 : 8;
            // Wt is always a power-of-two (pre_sort_transform_tensor pads the last dim to the
            // next power-of-two ≥ 2×TILE_WIDTH before dispatching). MAX_DEST_TILES is also a
            // power-of-two (4 or 8), so Wt % SUB_BLOCK_DIM == 0 is always satisfied.
            constexpr uint32_t SUB_BLOCK_DIM = (Wt < MAX_DEST_TILES) ? Wt : MAX_DEST_TILES;
            constexpr uint32_t NUM_SUB_BLOCKS = Wt / SUB_BLOCK_DIM;
            static_assert(Wt % SUB_BLOCK_DIM == 0, "Wt must be divisible by SUB_BLOCK_DIM");

            // Un-transpose sorted value tiles → input_tensor (Wt tiles).
            transpose_and_pack(
                input_tensor_transposed_dfb, input_tensor_dfb, Wt, /*prepare_uint16_value_for_pack=*/true);

            // Un-transpose sorted index tiles → rm_post_sort_index (Wt tiles).
            transpose_and_pack(index_tensor_transposed_dfb, rm_post_sort_index_dfb, Wt);

            // Untilize values: Wt tiles → TILE_HEIGHT RM pages in rm_value_output_dfb.
            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
            compute_kernel_hw_startup(dfb::input_tensor, dfb::index_tensor, dfb::rm_value_output);
            pack_untilize_init<SUB_BLOCK_DIM, Wt>(dfb::input_tensor, dfb::rm_value_output);
            input_tensor_dfb.wait_front(Wt);
            rm_value_output_dfb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, Wt>(dfb::input_tensor, 1, dfb::rm_value_output, b);
                input_tensor_dfb.pop_front(SUB_BLOCK_DIM);
            }
            rm_value_output_dfb.push_back(TILE_H);
            pack_untilize_uninit(dfb::rm_value_output);

            // Untilize indices: same chunked pack_untilize pattern but operating on the PACK-only
            // rm_post_sort_index_dfb
            // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
            compute_kernel_hw_startup(dfb::rm_post_sort_index, dfb::input_tensor, dfb::rm_index_output);
            pack_untilize_init<SUB_BLOCK_DIM, Wt>(dfb::rm_post_sort_index, dfb::rm_index_output);
            rm_post_sort_index_dfb.wait_front(Wt);
            rm_index_output_dfb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, Wt>(dfb::rm_post_sort_index, 1, dfb::rm_index_output, b);
                rm_post_sort_index_dfb.pop_front(SUB_BLOCK_DIM);
            }
            rm_index_output_dfb.push_back(TILE_H);
            pack_untilize_uninit(dfb::rm_index_output);
        }
#endif
    }  // Ht loop
}
