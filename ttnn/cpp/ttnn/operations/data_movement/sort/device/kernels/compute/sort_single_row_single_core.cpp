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
#include "api/dataflow/circular_buffer.h"

#include "sort_common.hpp"

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
    const uint32_t core_loop_count = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_transposed_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_transposed_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t index_tensor_output_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr bool descending = get_compile_time_arg_val(7);
    constexpr bool stable =
        get_compile_time_arg_val(8);  // TODO: In the future change LLK to have the option or add additional step with
                                      // checking values and indexes after the sorting
                                      // Issue: https://github.com/tenstorrent/tt-metal/issues/20625
    constexpr uint32_t synchronization_cb_id = get_compile_time_arg_val(9);
    constexpr bool is_row_major = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t rm_input_cb_id = get_compile_time_arg_val(11);
    constexpr uint32_t rm_value_output_cb_id = get_compile_time_arg_val(12);
    constexpr uint32_t rm_index_output_cb_id = get_compile_time_arg_val(13);
    // PACK-only CB that holds the un-transposed sorted index tiles between the
    // sort phase and the index pack_untilize_block.  Using a separate CB here
    // (rather than reusing index_tensor_cb_id, which is BRISC-pushed by the
    // writer) avoids a mixed-producer counter race: BRISC uses += into the L1
    // receive counter, while PACK overwrites it with PACK's own local counter,
    // so PACK's pushes silently clobber BRISC's pushes and cb_wait_front spins
    // forever.
    constexpr uint32_t rm_post_sort_index_cb_id = get_compile_time_arg_val(14);

    CircularBuffer input_tensor_cb(input_tensor_cb_id);
    CircularBuffer index_tensor_cb(index_tensor_cb_id);
    CircularBuffer input_tensor_transposed_cb(input_tensor_transposed_cb_id);
    CircularBuffer index_tensor_transposed_cb(index_tensor_transposed_cb_id);
    CircularBuffer value_tensor_cb(value_tensor_cb_id);
    CircularBuffer index_tensor_output_cb(index_tensor_output_cb_id);
    CircularBuffer synchronization_cb(synchronization_cb_id);
    CircularBuffer rm_input_cb(rm_input_cb_id);
    CircularBuffer rm_value_output_cb(rm_value_output_cb_id);
    CircularBuffer rm_index_output_cb(rm_index_output_cb_id);
    CircularBuffer rm_post_sort_index_cb(rm_post_sort_index_cb_id);

    constexpr uint32_t one_tile = 1;

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    // For TILE path: one-time initialisation before the loop.
    // For ROW_MAJOR: tilize/untilize are interleaved with sort each iteration.
    //
    // ROW_MAJOR requires compute_kernel_hw_startup to be called once before any
    // other compute API.  This initialises the MATH-PACK DST semaphore (via
    // llk_math_pack_sync_init + llk_pack_dest_init) so that tilize_block's
    // internal llk_math_wait_for_dest_available() does not spin forever.
    // Without this call the kernel deadlocks on the first tilize_block invocation.
    if constexpr (is_row_major) {
        compute_kernel_hw_startup(rm_input_cb_id, index_tensor_cb_id, input_tensor_cb_id);
    } else {
        compute_kernel_hw_startup(input_tensor_cb_id, input_tensor_transposed_cb_id);
        ckernel::topk_tile_init();
        transpose_init(input_tensor_cb_id);
    }

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const bool ascending = !descending;

        // ------------------------------------------------------------------
        // ROW_MAJOR: tilize one tile-row of RM data → Wt TILE-format tiles,
        // then reinitialise for sort.
        //
        // tilize_block uses out-of-order pack (llk_pack<true>) which writes
        // tiles to the CB WITHOUT calling cb_push_back internally.  The
        // explicit cb_reserve_back / cb_push_back pair is therefore required:
        //   • cb_reserve_back: guarantees write slots are free before the
        //     out-of-order pack writes to them.
        //   • cb_push_back:    signals the tiles as ready so that the
        //     subsequent sort_Wt_tiles_row_to_bitonic_sequence (which calls
        //     cb_wait_front) does not deadlock.
        // ------------------------------------------------------------------
        if constexpr (is_row_major) {
            constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT (tt::constants not available in device kernels)
            tilize_init(rm_input_cb_id, Wt, input_tensor_cb_id);
            rm_input_cb.wait_front(TILE_H);
            input_tensor_cb.reserve_back(Wt);
            tilize_block(rm_input_cb_id, Wt, input_tensor_cb_id);
            input_tensor_cb.push_back(Wt);
            rm_input_cb.pop_front(TILE_H);
            tilize_uninit(rm_input_cb_id, input_tensor_cb_id);

            // Re-initialise compute hardware for the sort phase.
            //
            // tilize_uninit does not reset the PACK side on WormholeB0 (the
            // Blackhole-only llk_pack_init path is skipped), so the packer is
            // still configured for the out-of-order tilize writes it just
            // performed.  binary_op_init_common (same critical calls as
            // compute_kernel_hw_startup: llk_math_pack_sync_init +
            // llk_pack_dest_init) re-arms the MATH-PACK DST semaphore and
            // resets PACK to normal mode so that pack_tile /
            // pack_reconfig_data_format inside sort_Wt_tiles_row_to_bitonic_sequence
            // work correctly.  Unlike compute_kernel_hw_startup this function
            // is safe to call multiple times per kernel invocation (same pattern
            // as layernorm_large_tensor.cpp's TILIZE_IN path).
            binary_op_init_common(input_tensor_cb_id, index_tensor_cb_id, input_tensor_transposed_cb_id);
            ckernel::topk_tile_init();
            transpose_init(input_tensor_cb_id);
        }

        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_cb,
            index_tensor_cb,
            input_tensor_transposed_cb,
            index_tensor_transposed_cb,
            Wt,
            /*switch_dir=*/true,
            ascending,
            /*end_phase(log2(K))=*/5);

        // Wait for bitonic sequence of Wt tiles
        input_tensor_transposed_cb.wait_front(Wt);
        index_tensor_transposed_cb.wait_front(Wt);

        // Sort and merge step of bitonic merge sort
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }

        synchronization_cb.reserve_back(one_tile);
        synchronization_cb.push_back(one_tile);

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

                        synchronization_cb.wait_front(one_tile);
                        synchronization_cb.pop_front(one_tile);
                        synchronization_cb.reserve_back(one_tile);

                        reconfig_data_format_srca(input_tensor_transposed_cb_id, index_tensor_transposed_cb_id);
                        copy_init(index_tensor_transposed_cb_id);
                        copy_tile(index_tensor_transposed_cb_id, left_tile_id, index_dest_start);
                        copy_tile(index_tensor_transposed_cb_id, right_tile_id, index_dest_end);

                        reconfig_data_format_srca(index_tensor_transposed_cb_id, input_tensor_transposed_cb_id);
                        copy_init(input_tensor_transposed_cb_id);
                        copy_tile(input_tensor_transposed_cb_id, left_tile_id, input_dest_start);
                        copy_tile(input_tensor_transposed_cb_id, right_tile_id, input_dest_end);

                        uint32_t tile_input_low = input_dest_start;
                        uint32_t tile_input_high = input_dest_end;
                        uint32_t tile_index_low = index_dest_start;
                        uint32_t tile_index_high = index_dest_end;

                        if (sub == 1) {
                            // Use sort LLK only the last stage to sort the last pair of tiles - speed up
                            ckernel::topk_local_sort(/*idst=*/0, (int)dir, /*end_phase(log2(K))=*/5);
                        } else {
                            ckernel::topk_merge(/*idst=*/0, m_iter, /*k=*/64);

                            if (dir) {
                                // topk_merge puts smallest values in DEST[0] and largest in DEST[1]
                                // We swap their indices when using descending order
                                tile_input_low = input_dest_end;
                                tile_input_high = input_dest_start;
                                tile_index_low = index_dest_end;
                                tile_index_high = index_dest_start;
                            }
                        }

                        tile_regs_commit();
                        tile_regs_wait();

                        pack_reconfig_data_format(input_tensor_transposed_cb_id);
                        pack_tile<true>(tile_input_low, input_tensor_transposed_cb_id, left_tile_id);
                        pack_tile<true>(tile_input_high, input_tensor_transposed_cb_id, right_tile_id);

                        pack_reconfig_data_format(index_tensor_transposed_cb_id);
                        pack_tile<true>(tile_index_low, index_tensor_transposed_cb_id, left_tile_id);
                        pack_tile<true>(tile_index_high, index_tensor_transposed_cb_id, right_tile_id);

                        synchronization_cb.push_back(one_tile);

                        tile_regs_release();
                    }
                }
            }
        }

        synchronization_cb.wait_front(one_tile);
        synchronization_cb.pop_front(one_tile);

        input_tensor_transposed_cb.reserve_back(Wt);
        index_tensor_transposed_cb.reserve_back(Wt);

        input_tensor_transposed_cb.pop_front(Wt);
        index_tensor_transposed_cb.pop_front(Wt);

        input_tensor_transposed_cb.push_back(Wt);
        index_tensor_transposed_cb.push_back(Wt);

        // TILE path: transpose-and-pack to 2-tile streaming CBs so the reader
        // and writer can stream tiles one-by-one to DRAM (existing behaviour).
        //
        // ROW_MAJOR path: transpose-and-pack to the now-empty Wt-tile CBs
        // (input_tensor_cb_id for values, index_tensor_cb_id for
        // indices), then untilize the full row back to RM pages.
        if constexpr (!is_row_major) {
            // Values tensor → 2-tile streaming CB (writer drains to DRAM)
            transpose_and_pack(input_tensor_transposed_cb, value_tensor_cb, Wt);
            // Index tensor → 2-tile streaming CB (reader drains to DRAM)
            transpose_and_pack(index_tensor_transposed_cb, index_tensor_output_cb, Wt);
        } else {
            constexpr uint32_t TILE_H = 32;

            constexpr uint32_t MAX_DEST_TILES = DST_ACCUM_MODE ? 4 : 8;
            // Wt is always a power-of-two (pre_sort_transform_tensor pads the last dim to the
            // next power-of-two ≥ 2×TILE_WIDTH before dispatching).  MAX_DEST_TILES is also a
            // power-of-two (4 or 8), so Wt % SUB_BLOCK_DIM == 0 is always satisfied.
            constexpr uint32_t SUB_BLOCK_DIM = (Wt < MAX_DEST_TILES) ? Wt : MAX_DEST_TILES;
            constexpr uint32_t NUM_SUB_BLOCKS = Wt / SUB_BLOCK_DIM;
            static_assert(Wt % SUB_BLOCK_DIM == 0, "Wt must be divisible by SUB_BLOCK_DIM");

            // Un-transpose sorted value tiles → input_tensor_cb_id (Wt tiles).
            transpose_and_pack(input_tensor_transposed_cb, input_tensor_cb, Wt);

            // Un-transpose sorted index tiles → rm_post_sort_index_cb_id (Wt tiles).
            transpose_and_pack(index_tensor_transposed_cb, rm_post_sort_index_cb, Wt);

            // Untilize values: Wt tiles → TILE_HEIGHT RM pages in rm_value_output_cb.
            binary_op_init_common(input_tensor_cb_id, index_tensor_cb_id, rm_value_output_cb_id);
            pack_untilize_init<SUB_BLOCK_DIM, Wt>(input_tensor_cb_id, rm_value_output_cb_id);
            input_tensor_cb.wait_front(Wt);
            rm_value_output_cb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, Wt>(input_tensor_cb_id, 1, rm_value_output_cb_id, b);
                input_tensor_cb.pop_front(SUB_BLOCK_DIM);
            }
            rm_value_output_cb.push_back(TILE_H);
            pack_untilize_uninit(rm_value_output_cb_id);

            // Untilize indices: same chunked pack_untilize pattern but operating on the PACK-only rm_post_sort_index_cb
            binary_op_init_common(rm_post_sort_index_cb_id, input_tensor_cb_id, rm_index_output_cb_id);
            pack_untilize_init<SUB_BLOCK_DIM, Wt>(rm_post_sort_index_cb_id, rm_index_output_cb_id);
            rm_post_sort_index_cb.wait_front(Wt);
            rm_index_output_cb.reserve_back(TILE_H);
            for (uint32_t b = 0; b < NUM_SUB_BLOCKS; ++b) {
                pack_untilize_block<SUB_BLOCK_DIM, Wt>(rm_post_sort_index_cb_id, 1, rm_index_output_cb_id, b);
                rm_post_sort_index_cb.pop_front(SUB_BLOCK_DIM);
            }
            rm_index_output_cb.push_back(TILE_H);
            pack_untilize_uninit(rm_index_output_cb_id);
        }
    }  // Ht loop
}
