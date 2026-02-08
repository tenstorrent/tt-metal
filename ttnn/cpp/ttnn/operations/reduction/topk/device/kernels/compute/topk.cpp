// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

/**
 * Transpose tiles from width-height to height-width format and pack to destination buffer
 * Used in final stage to convert sorted results back to output format
 *
 * @param input_cb_index    Circular buffer index containing tiles to transpose (source buffer)
 * @param dest_cb_index     Circular buffer index to store transposed tiles (destination buffer)
 * @param total_tiles       Number of tiles to process and transpose
 */
FORCE_INLINE void transpose_and_pack(
    const uint32_t input_cb_index, const uint32_t dest_cb_index, const uint32_t total_tiles) {
    // Configure data formats for transpose operation
    reconfig_data_format_srca(input_cb_index);
    transpose_wh_init_short(input_cb_index);
    pack_reconfig_data_format(input_cb_index);

    // Wait for all tiles to be available (double-buffered, hence 2 * total_tiles)
    cb_wait_front(input_cb_index, 2 * total_tiles);
    for (uint32_t i = 0; i < total_tiles; ++i) {
        acquire_dst();
        cb_reserve_back(dest_cb_index, 1);

        // Transpose tile from WH to HW format
        transpose_wh_tile(input_cb_index, i, 0);

        // Pack transposed tile to destination
        pack_tile(0, dest_cb_index);
        cb_push_back(dest_cb_index, 1);
        release_dst();
    }  // i loop
    cb_pop_front(input_cb_index, 2 * total_tiles);
}

/**
 * Pack two destination tiles (values and indices) to their respective circular buffers
 * Used after merge operation to store sorted results
 *
 * @param cb0            Circular buffer index for packing the first tile (values)
 * @param cb1            Circular buffer index for packing the second tile (indices)
 * @param base_offset    Base offset in destination registers (first tile at base_offset, second at base_offset+1)
 */
FORCE_INLINE void pack_results(const uint32_t cb0, const uint32_t cb1, const uint32_t base_offset) {
    // Pack first tile (values) to cb0
    pack_reconfig_data_format(cb0);
    pack_tile(base_offset, cb0);

    // Pack second tile (indices) to cb1, reconfig only if different buffer
    if (cb0 != cb1) {
        pack_reconfig_data_format(cb1);
    }
    pack_tile(base_offset + 1, cb1);
}

/**
 * Read tiles from circular buffer and transpose them to destination registers
 * Used to prepare input tiles for sorting operations
 *
 * @param cb               Circular buffer index to read tiles from
 * @param base_offset      Base offset in destination registers where transposed tiles will be stored
 * @param get_two         Boolean flag: true to transpose two tiles (tiles 0,1 -> dest base_offset, base_offset+1),
 *                        false to transpose only one tile (tile 0 -> dest base_offset)
 */
FORCE_INLINE void read_cb_and_transpose(const uint32_t cb, const uint32_t base_offset, const bool get_two = true) {
    reconfig_data_format_srca(cb);
    transpose_wh_init_short(cb);

    // Transpose first tile to destination register
    transpose_wh_tile(cb, 0, base_offset + 0);
    if (get_two) {
        // Transpose second tile if processing two tiles at once (initial iteration)
        transpose_wh_tile(cb, 1, base_offset + 1);
    }
}

/**
 * Utility function: Wait for tiles and pop them from front of circular buffer
 * Refactored to reduce code duplication
 *
 * @param cb      Circular buffer index to operate on
 * @param count   Number of tiles to wait for and then remove from the front of the buffer
 */
FORCE_INLINE void cb_wait_pop_front(const uint32_t cb, const uint32_t count) {
    cb_wait_front(cb, count);
    cb_pop_front(cb, count);
}

/**
 * Utility function: Reserve space and push tiles to back of circular buffer
 * Refactored to reduce code duplication
 *
 * @param cb      Circular buffer index to operate on
 * @param count   Number of tile slots to reserve at the back and then mark as available
 */
FORCE_INLINE void cb_reserve_push_back(const uint32_t cb, const uint32_t count) {
    cb_reserve_back(cb, count);
    cb_push_back(cb, count);
}

void kernel_main() {
    // Runtime arguments
    const uint32_t work_per_core = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t input_val_cb_index = get_compile_time_arg_val(0);        // Input values circular buffer
    constexpr uint32_t input_ind_cb_index = get_compile_time_arg_val(1);        // Input indices circular buffer
    constexpr uint32_t transposed_val_cb_index = get_compile_time_arg_val(2);   // Transposed values buffer
    constexpr uint32_t transposed_ind_cb_index = get_compile_time_arg_val(3);   // Transposed indices buffer
    constexpr uint32_t result_prep_val_cb_index = get_compile_time_arg_val(4);  // Result preparation values buffer
    constexpr uint32_t result_prep_ind_cb_index = get_compile_time_arg_val(5);  // Result preparation indices buffer
    constexpr uint32_t output_val_cb_index = get_compile_time_arg_val(6);       // Final output values buffer
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(7);       // Final output indices buffer
    constexpr uint32_t Ht = get_compile_time_arg_val(8);                        // Height in tiles
    constexpr uint32_t Wt = get_compile_time_arg_val(9);                        // Width in tiles
    constexpr uint32_t output_tiles = get_compile_time_arg_val(10);             // Number of output tiles (ceil(K/32))
    constexpr uint32_t largest = get_compile_time_arg_val(11);                  // 1 for largest K, 0 for smallest K

    // Initialize kernel components
    ckernel::topk_tile_init();
    transpose_wh_init(input_val_cb_index, output_val_cb_index);
    transpose_wh_init(input_ind_cb_index, output_ind_cb_index);

    constexpr int end_phase = 5;  // The end phase of the local sort, based on topk_local_sort documentation
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        uint32_t ktiles_saved = 0;
        /*
        ================================================================================================
        TOPK ALGORITHM IMPLEMENTATION - INSERTION SORT WITH DOUBLE BUFFERING
        ================================================================================================

        This kernel implements TopK using an insertion sort algorithm that maintains a sliding window
        of the K largest/smallest elements across all input tiles. The algorithm processes input
        tiles row by row (height dimension) and maintains sorted results in a double-buffered format.

        KEY CONCEPTS:
        - output_tiles = ceil(K/32): Number of tiles needed to store K elements
        - Each tile contains 32 elements, so for K=128, we need output_tiles=4
        - Double buffering: result_prep buffers have size 2*output_tiles for efficient insertion
        - ktiles_saved: Tracks how many output tiles currently contain valid sorted data

        ALGORITHM PHASES:

        PHASE 1: INITIALIZATION (First iteration, count=1)
        - Read first TWO input tiles (2 x values + 2 x indices)
        - Transpose from WH to HW format for processing
        - Sort these 64 elements using topk_local_sort
        - Store sorted results in result_prep buffer
        - Set ktiles_saved = 2 (consumed 2 tiles worth of sorted data)

        PHASE 2: INCREMENTAL INSERTION (count=2 to Wt-1)
        - Read ONE input tile at a time (32 new elements)
        - For each of the output_tiles positions, perform insertion sort:
          * Compare new tile with existing sorted tile at that position
          * Merge using topk_local_sort (keeping K largest/smallest)
          * Store result back to result_prep buffer

        PHASE 3: BUFFER MANAGEMENT STRATEGY
        Three distinct cases based on current state:

        Case A: FIRST SORT (ktiles_saved == 0)
        - Process initial two transposed tiles
        - No existing sorted data to merge with
        - Increment: output_tiles (jump to next buffer half)
        - Result: ktiles_saved = 2

        Case B: GROWING PHASE (index >= ktiles_saved-1 && ktiles_saved < output_tiles)
        - Still building up the sorted buffer
        - Insert new tile at the next available position
        - Each insertion increases ktiles_saved by 1
        - Increment: output_tiles - index (adaptive positioning)

        Case C: STEADY STATE (ktiles_saved == output_tiles)
        - Buffer is full with K best elements
        - New tiles compete with existing worst elements
        - Only better elements replace existing ones
        - Increment: 1 (simple linear processing)

        PHASE 4: DOUBLE BUFFERING MECHANICS
        - result_prep buffer alternates between two halves of size output_tiles
        - After each insertion, we flip between reading from one half and writing to the other
        - This prevents data corruption during the merge operation
        - The 'incr' variable controls how far to advance buffer pointers

        PHASE 5: OUTPUT GENERATION
        - After processing all Wt input tiles, result_prep contains the K best elements
        - Transpose results back from HW to WH format
        - Pack final results into output circular buffers

        EXAMPLE (K=128, output_tiles=4):
        Initial: [ ][ ][ ][ ] (4 empty slots)
        After 2 tiles: [64 elements][sorted][ ][ ] (ktiles_saved=2)
        After 3 tiles: [32 elements][32 elements][32 elements][ ] (ktiles_saved=3)
        After 4 tiles: [32][32][32][32] (ktiles_saved=4, buffer full)
        Subsequent tiles compete with existing elements, replacing worse ones
        */

        // Main processing loop: refactored into single loop to fit TRISC2 memory constraints
        uint32_t input_take = 2;  // First iteration processes 2 tiles, subsequent iterations process 1
        for (uint32_t count = 1; count < Wt; count++) {
            // Read input tiles (2 on first iteration, 1 on subsequent) and transpose to HW format
            // Wait for input tiles to become available
            cb_wait_front(input_val_cb_index, input_take);
            cb_wait_front(input_ind_cb_index, input_take);

            // Reserve space in transposed buffers for processing
            cb_reserve_back(transposed_val_cb_index, input_take);
            cb_reserve_back(transposed_ind_cb_index, input_take);

            acquire_dst();

            // Transpose input tiles from WH to HW format and pack to intermediate buffers
            read_cb_and_transpose(input_val_cb_index, 0, (2 == input_take));  // Values: dest regs 0,1
            read_cb_and_transpose(input_ind_cb_index, 2, (2 == input_take));  // Indices: dest regs 2,3

            // Pack transposed values
            pack_reconfig_data_format(transposed_val_cb_index);
            pack_tile(0, transposed_val_cb_index);
            if (input_take == 2) {
                pack_tile(1, transposed_val_cb_index);  // Pack second tile on first iteration
            }

            // Pack transposed indices
            pack_reconfig_data_format(transposed_ind_cb_index);
            pack_tile(2, transposed_ind_cb_index);
            if (input_take == 2) {
                pack_tile(3, transposed_ind_cb_index);  // Pack second tile on first iteration
            }

            release_dst();

            // Remove processed tiles from input buffers
            cb_pop_front(input_val_cb_index, input_take);
            cb_pop_front(input_ind_cb_index, input_take);

            // Store transposed tiles for insertion sort processing
            cb_push_back(transposed_val_cb_index, input_take);
            cb_push_back(transposed_ind_cb_index, input_take);

            // Insertion sort into result preparation buffer
            // Process each output tile position for insertion sort
            for (uint32_t index = 0; index < output_tiles; index++) {
                // Initialize variables for current insertion iteration
                uint32_t incr = 1;                        // Default buffer advance increment
                uint32_t transposed_offset = 0;           // Offset in transposed buffer (0 or 1)
                uint32_t cb0 = result_prep_val_cb_index;  // Source buffer for values
                uint32_t cb1 = result_prep_ind_cb_index;  // Source buffer for indices
                uint32_t cb2 = transposed_val_cb_index;   // Destination buffer for values
                uint32_t cb3 = transposed_ind_cb_index;   // Destination buffer for indices
                uint32_t in_cb_offset = incr;             // Input buffer offset

                // CASE A: FIRST SORT - Process initial two tiles
                if (ktiles_saved == 0) {
                    incr = output_tiles;             // Jump to next buffer half
                    transposed_offset = 1;           // Use second transposed tile
                    cb0 = transposed_val_cb_index;   // Source: transposed values
                    cb1 = transposed_ind_cb_index;   // Source: transposed indices
                    cb2 = result_prep_val_cb_index;  // Dest: result prep values
                    cb3 = result_prep_ind_cb_index;  // Dest: result prep indices
                    ktiles_saved += 2;               // Mark 2 tiles as processed
                    index = output_tiles;            // Exit loop after this iteration
                    in_cb_offset = input_take;       // Use both input tiles

                    // CASE B: GROWING PHASE - Buffer not yet full, insert at next position
                } else if ((index >= (ktiles_saved - 1)) && (ktiles_saved < output_tiles)) {
                    incr = output_tiles - index;     // Adaptive buffer positioning
                    ktiles_saved++;                  // Increment saved tile count
                    index = output_tiles;            // Exit loop after this iteration
                    cb2 = result_prep_val_cb_index;  // Store result back to prep buffer
                    cb3 = result_prep_ind_cb_index;
                    in_cb_offset = incr;

                    // CASE C: STEADY STATE - Buffer full, compete with existing elements
                    // (Uses default values set above - simple linear processing)
                }

                // Prepare data for merge operation
                // Wait for required tiles to be available
                cb_wait_front(cb0, in_cb_offset);  // Wait for existing sorted data
                cb_wait_front(cb1, in_cb_offset);
                if (transposed_offset == 0) {
                    cb_wait_front(transposed_val_cb_index, 1);  // Wait for new input tile
                    cb_wait_front(transposed_ind_cb_index, 1);
                }

                // Reserve space for intermediate results
                cb_reserve_back(transposed_val_cb_index, 1);
                cb_reserve_back(transposed_ind_cb_index, 1);

                acquire_dst();

                // Load tiles into destination registers for merging
                // Load existing sorted values into dest reg 0
                copy_tile_to_dst_init_short_with_dt(cb1, cb0);
                copy_tile(cb0, 0, 0);

                // Load existing sorted indices into dest reg 2
                copy_tile_to_dst_init_short_with_dt(cb0, cb1);
                copy_tile(cb1, 0, 2);

                // Load new input values into dest reg 1
                copy_tile_to_dst_init_short_with_dt(transposed_ind_cb_index, transposed_val_cb_index);
                copy_tile(transposed_val_cb_index, transposed_offset, 1);

                // Load new input indices into dest reg 3
                copy_tile_to_dst_init_short_with_dt(transposed_val_cb_index, transposed_ind_cb_index);
                copy_tile(transposed_ind_cb_index, transposed_offset, 3);

                // Perform merge and sort operation
                // Merge and sort 64 elements (32 existing + 32 new) using topk_local_sort
                // Results: dest reg 0 = top 32 elements, dest reg 1 = bottom 32 elements
                // largest flag determines ascending (0) vs descending (1) sort order
                ckernel::topk_local_sort(0, (int)!largest, end_phase);

                // Store sorted results back to buffers
                // Reserve space for storing the best K elements
                cb_reserve_back(result_prep_val_cb_index, incr);
                cb_reserve_back(result_prep_ind_cb_index, incr);

                // Pack sorted results: dest reg 0 -> result buffer, dest reg 1 -> secondary buffer
                pack_results(result_prep_val_cb_index, cb2, 0);  // Store top 32 elements
                pack_results(result_prep_ind_cb_index, cb3, 2);  // Store corresponding indices

                // Clean up source buffers
                cb_pop_front(cb0, in_cb_offset);
                cb_pop_front(cb1, in_cb_offset);

                // Advance result prep buffer pointers
                cb_push_back(result_prep_val_cb_index, incr);
                cb_push_back(result_prep_ind_cb_index, incr);

                release_dst();

                // Clean up transposed buffers if we consumed from them
                if (transposed_offset == 0) {
                    cb_pop_front(transposed_val_cb_index, 1);
                    cb_pop_front(transposed_ind_cb_index, 1);
                }

                // Maintain transposed buffer structure
                cb_push_back(transposed_val_cb_index, 1);
                cb_push_back(transposed_ind_cb_index, 1);
            }  // index loop

            // After first iteration, process only 1 tile at a time
            input_take = 1;

            // Clean up intermediate transposed tile
            cb_wait_pop_front(transposed_val_cb_index, 1);
            cb_wait_pop_front(transposed_ind_cb_index, 1);
        }  // count loop

        // Final output preparation
        // Prepare result buffers for final output
        cb_reserve_push_back(result_prep_val_cb_index, output_tiles);
        cb_reserve_push_back(result_prep_ind_cb_index, output_tiles);

        // Transpose and pack final results to output buffers
        // Convert sorted results from HW back to WH format for output
        transpose_and_pack(result_prep_val_cb_index, output_val_cb_index, output_tiles);
        transpose_and_pack(result_prep_ind_cb_index, output_ind_cb_index, output_tiles);
    }  // core_loop loop
}
