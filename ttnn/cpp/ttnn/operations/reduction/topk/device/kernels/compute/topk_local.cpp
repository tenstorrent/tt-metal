// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

#include "topk_common_funcs.hpp"


/**
 * TopK Multicore Compute Kernel Implementation - Local Processing Phase
 *
 * This kernel implements the first stage of a two-stage multicore TopK algorithm that uses
 * width-based parallelization with bitonic sorting for optimal hardware utilization.
 *
 * ================================================================================================
 * MULTICORE TOPK ALGORITHM IMPLEMENTATION - DIVIDE-AND-CONQUER WITH BITONIC SORTING
 * ================================================================================================
 *
 * OVERVIEW:
 * The multicore TopK implementation splits the workload across multiple cores by dividing
 * the input tensor along the width dimension. Each core processes its assigned width chunk
 * independently using bitonic sorting, then sends results to a final aggregation core.
 *
 * ALGORITHM PHASES:
 *
 * PHASE 1: WORK DISTRIBUTION (Program Factory Level)
 * - Input tensor width is divided among available cores
 * - Each local core gets a contiguous chunk of width tiles (Wt_local)
 * - One final core is designated for global aggregation
 * - Core configuration optimized based on K value and available L1 memory
 *
 * PHASE 2: LOCAL PROCESSING (This Kernel - topk_local.cpp)
 * - Each core independently processes its width chunk using bitonic sort
 * - Bitonic sort naturally handles parallel merge operations
 * - Results in locally sorted TopK values and indices for each chunk
 * - Outputs Kt tiles (ceil(K/32)) of sorted data per height row
 *
 * PHASE 3: GLOBAL AGGREGATION (topk_final.cpp)
 * - Final core receives Kt tiles from each local core
 * - Performs final bitonic merge across all received chunks
 * - Produces globally optimal TopK results
 *
 * BITONIC SORTING STRATEGY:
 *
 * 1. INITIAL LOCAL SORT:
 * - Process input tiles in pairs (2 tiles = 64 elements at a time)
 * - Transpose from WH to HW format for optimal processing
 * - Apply topk_local_sort to create locally sorted sequences
 * - Alternate sort direction for bitonic properties
 *
 * 2. ITERATIVE DIVIDE-AND-CONQUER:
 * - log(Wt_local) iterations of bitonic merge
 * - Each iteration doubles the sequence length being compared
 * - Maintains bitonic properties through directional alternation
 * - In-place operations minimize memory overhead
 *
 * 3. RESULT EXTRACTION:
 * - Extract top Kt tiles containing the K best elements
 * - Transpose back to WH format for output
 * - Send to aggregation core via semaphore-synchronized communication
 *
 * MEMORY MANAGEMENT:
 * - Double-buffered input for continuous data flow
 * - Single-buffered intermediate results for in-place operations
 * - Separate circular buffers for values and indices
 * - Optimized buffer sizes based on L1 memory constraints
 *
 * INTER-CORE COMMUNICATION:
 * - Semaphore-based synchronization between local and final cores
 * - Direct NoC transfers for efficient data movement
 * - Flow control to prevent buffer overflow
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Time complexity: O(Wt_local * log²(Wt_local) + K*log(num_cores))
 * - Space complexity: O(Wt_local + K) per core
 * - Scales efficiently with number of available cores
 * - Memory bandwidth optimized through tiled processing
 *
 * EXAMPLE WORKFLOW (K=128, 4 local cores):
 * Core 0: Processes tiles [0-15]   → Local TopK(128) → Send to final core
 * Core 1: Processes tiles [16-31]  → Local TopK(128) → Send to final core
 * Core 2: Processes tiles [32-47]  → Local TopK(128) → Send to final core
 * Core 3: Processes tiles [48-63]  → Local TopK(128) → Send to final core
 * Final:  Receives 4×128 elements → Global TopK(128) → Output final result
 */

void kernel_main() {
    // Compiletime args
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t Kt = get_compile_time_arg_val(9);
    constexpr uint32_t logk = get_compile_time_arg_val(10);
    constexpr uint32_t logWt = get_compile_time_arg_val(11);
    constexpr uint32_t largest = get_compile_time_arg_val(12);
    constexpr uint32_t sorted = get_compile_time_arg_val(13);

    // Runtime args
    uint32_t direction_init = get_arg_val<uint32_t>(0);

    // Constants
    // Dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    constexpr uint32_t tiles_per_seq = (K + 31) / 32;

    // Supports K only up to 64
    int end_phase = (K <= 64) ? logk - 1 : 5;

    ckernel::topk_tile_init();
    transpose_wh_init(input_cb_index, input_transposed_cb_index);
    transpose_wh_init(index_cb_index, index_transposed_cb_index);

    bool switch_dir = (K == 64);
    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);

    // Process each height row independently
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = !largest;  // Sort direction for bitonic sequence properties

        // Initial bitonic sort on local width chunk
        process_and_sort_tiles(
            input_cb_index,             // Input values buffer (double-buffered)
            index_cb_index,             // Input indices buffer (double-buffered)
            input_transposed_cb_index,  // Transposed values staging buffer
            index_transposed_cb_index,  // Transposed indices staging buffer
            Wt,                         // Width tiles for this local chunk
            switch_dir,                 // Whether to alternate sort direction
            ascending,                  // Current sort direction
            end_phase);                 // Ending phase for local sort

        uint32_t num_k_sequences = (Wt * 32) / K;  // Number of K-element sequences in chunk

        // Iterative bitonic sort across the entire local width chunk
        // Perform log(Wt) iterations of divide-and-conquer merging:
        // - Iteration 0: Compare tiles (0,1), (2,3), (4,5), ... → pairs of 64 elements
        // - Iteration 1: Compare tiles (0,2), (4,6), (8,10), ... → groups of 128 elements
        // - Iteration n: Compare tiles with distance 2^n → groups of 64*(2^(n+1)) elements
        // Final iteration produces locally sorted TopK results for this width chunk.
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            process_iteration(
                m_iter,                     // Current merge iteration (0 to logWt-1)
                K,                          // TopK value (number of elements to find)
                Wt,                         // Width tiles in local chunk
                num_k_sequences,            // Number of K-element sequences (updated each iter)
                tiles_per_seq,              // Tiles per sequence (ceil(K/32))
                input_transposed_cb_index,  // Values buffer for in-place operations
                index_transposed_cb_index,  // Indices buffer for in-place operations
                input_dest_start,           // Destination register 0 (first tile)
                input_dest_end,             // Destination register 1 (second tile)
                index_dest_start,           // Destination register 2 (first indices)
                index_dest_end,             // Destination register 3 (second indices)
                !direction_init,            // Base sort direction
                switch_dir,                 // Whether to switch direction per iteration
                logk,                       // log2(K) for bitonic network depth
                seq_per_2tiles,             // Sequences that fit in 2 tiles
                largest);                   // Find largest (true) or smallest (false)
        }  // m_iter loop

        // Extract and prepare local TopK results for transmission
        // After bitonic merging, the top Kt tiles contain the locally optimal
        // TopK elements. Extract these and prepare for sending to the final core.

        // Configure data formats for tile copying and prepare value tiles
        reconfig_data_format_srca(input_transposed_cb_index);
        copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
        pack_reconfig_data_format(input_transposed_cb_index);

        // Extract local TopK values (first Kt tiles contain best values)
        cb_wait_front(input_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(values_cb_index, 1);
            copy_tile(input_transposed_cb_index, i, 0);  // Copy i-th sorted value tile
            pack_tile(0, values_cb_index);               // Pack for output transmission
            cb_push_back(values_cb_index, 1);
            release_dst();
        }
        // Clean up remaining tiles in transposed buffer
        cb_wait_front(input_transposed_cb_index, Wt);
        cb_pop_front(input_transposed_cb_index, Wt);

        // Extract local TopK indices (corresponding to the best values)
        reconfig_data_format_srca(index_transposed_cb_index);
        copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        cb_wait_front(index_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(output_ind_cb_index, 1);
            copy_tile(index_transposed_cb_index, i, 0);  // Copy i-th sorted index tile
            pack_tile(0, output_ind_cb_index);           // Pack for output transmission
            cb_push_back(output_ind_cb_index, 1);
            release_dst();
        }
        // Clean up remaining tiles in transposed buffer
        cb_wait_front(index_transposed_cb_index, Wt);
        cb_pop_front(index_transposed_cb_index, Wt);

        // NOTE: At this point, values_cb_index and output_ind_cb_index contain
        // the locally optimal TopK results for this core's width chunk.
        // The writer kernel will send these to the final aggregation core.
    }  // ht loop
}
