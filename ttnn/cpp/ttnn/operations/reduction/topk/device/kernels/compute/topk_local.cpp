// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

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
 * PHASE 3: TREE MERGE (this kernel together with writer_local_topk.cpp)
 * - log2(num_local_cores) rounds; in round r core i with i % 2^(r+1) == 0 receives the Kt tiles of
 *   core i + 2^r next to its own and keeps the top Kt of the pair with one bitonic merge step
 * - The kept sequence's sort direction alternates so the partners of the next round are bitonic
 *
 * PHASE 4: GLOBAL AGGREGATION (topk_final.cpp)
 * - The tree survivor(s) send their Kt tiles to the final core
 * - With the tree run to the root the final core is a pass through (Wt_final == Kt)
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
 * - Semaphore-based synchronization between local cores (credit/data per tree round) and with the final core
 * - Direct NoC transfers for efficient data movement
 * - Flow control to prevent buffer overflow
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Time complexity: O(Wt_local * log²(Wt_local) + K*log(num_cores))
 * - Space complexity: O(Wt_local + K) per core
 * - Scales efficiently with number of available cores
 * - Memory bandwidth optimized through tiled processing
 *
 * EXAMPLE WORKFLOW (K=64, 4 local cores):
 * Core 0: Processes tiles [0-15]   → Local TopK(64) → merges core 1's, then core 2's → Send to final core
 * Core 1: Processes tiles [16-31]  → Local TopK(64) → Send to core 0 (round 0)
 * Core 2: Processes tiles [32-47]  → Local TopK(64) → merges core 3's → Send to core 0 (round 1)
 * Core 3: Processes tiles [48-63]  → Local TopK(64) → Send to core 2 (round 0)
 * Final:  Receives 64 elements → Output final result
 */

// Copies the first num_tiles tiles of src_dfb_index into dst_dfb_index one tile at a time, packing with the
// destination CB format, then releases src_pop tiles from the source.
void copy_front_tiles(
    std::uint32_t src_dfb_index, std::uint32_t dst_dfb_index, std::uint32_t num_tiles, std::uint32_t src_pop) {
    DataflowBuffer src_dfb(static_cast<uint16_t>(src_dfb_index));
    DataflowBuffer dst_dfb(static_cast<uint16_t>(dst_dfb_index));

    reconfig_data_format_srca(src_dfb_index);
    copy_init(src_dfb_index);
    pack_reconfig_data_format(dst_dfb_index);

    src_dfb.wait_front(static_cast<uint16_t>(num_tiles));
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(src_dfb_index, i, 0);
        tile_regs_commit();

        dst_dfb.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dst_dfb_index);
        tile_regs_release();

        dst_dfb.push_back(1);
    }
    src_dfb.wait_front(static_cast<uint16_t>(src_pop));
    src_dfb.pop_front(static_cast<uint16_t>(src_pop));
}

void kernel_main() {
    // Compile time args
    constexpr std::uint32_t input_dfb_index = get_compile_time_arg_val(0);
    constexpr std::uint32_t index_dfb_index = get_compile_time_arg_val(1);
    constexpr std::uint32_t input_transposed_dfb_index = get_compile_time_arg_val(2);
    constexpr std::uint32_t index_transposed_dfb_index = get_compile_time_arg_val(3);
    constexpr std::uint32_t values_dfb_index = get_compile_time_arg_val(4);
    constexpr std::uint32_t output_ind_dfb_index = get_compile_time_arg_val(5);
    constexpr std::uint32_t Ht = get_compile_time_arg_val(6);
    constexpr std::uint32_t Wt = get_compile_time_arg_val(7);
    constexpr std::uint32_t K = get_compile_time_arg_val(8);
    constexpr std::uint32_t Kt = get_compile_time_arg_val(9);
    constexpr std::uint32_t logk = get_compile_time_arg_val(10);
    constexpr std::uint32_t logWt = get_compile_time_arg_val(11);
    constexpr std::uint32_t largest = get_compile_time_arg_val(12);
    constexpr std::uint32_t sorted = get_compile_time_arg_val(13);
    constexpr bool stable_sort = get_compile_time_arg_val(14) == 1;  // Ties keep the lowest index

    // Fused-key stable mode: sort packed [bf16|u16] keys with the unstable network instead of
    // running the comparator-stable network on separate value/index tiles.
    constexpr bool fused_keys = get_compile_time_arg_val(15) == 1;
    // The packed key IS the stable tie-break; the network itself runs unstable in fused mode.
    constexpr bool network_stable = stable_sort && !fused_keys;
    // Tree merge: the writer lands [own Kt | partner Kt] tiles in the landing CBs; the merge CBs are the
    // 2*Kt-tile in-place workspace of the one bitonic merge step per round.
    constexpr std::uint32_t landing_values_dfb_index = get_compile_time_arg_val(16);
    constexpr std::uint32_t landing_indices_dfb_index = get_compile_time_arg_val(17);
    constexpr std::uint32_t merge_values_dfb_index = get_compile_time_arg_val(18);
    constexpr std::uint32_t merge_indices_dfb_index = get_compile_time_arg_val(19);

    // Runtime args
    std::uint32_t direction_init = get_arg_val<std::uint32_t>(0);
    const std::uint32_t core_id = get_arg_val<std::uint32_t>(1);          // Index among the local cores
    const std::uint32_t num_recv_rounds = get_arg_val<std::uint32_t>(2);  // Tree rounds this core receives in

    // Constants
    // Dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr std::uint32_t input_dest_start = 0;
    constexpr std::uint32_t index_dest_start = 2;
    constexpr std::uint32_t input_dest_end = 1;
    constexpr std::uint32_t index_dest_end = 3;
    constexpr std::uint32_t tiles_per_seq = (K + 31) / 32;

    // Supports K only up to 64
    const int end_phase = (K <= 64) ? logk - 1 : 5;

    compute_kernel_hw_startup(input_dfb_index, index_dfb_index, input_transposed_dfb_index);
    ckernel::topk_tile_init<fused_keys>();
    constexpr auto tie_order = ckernel::topk_tie_order_from_global_direction(largest != 0);

    const bool switch_dir = (K == 64);
    uint32_t seq_per_2tiles = std::max<uint32_t>((2 * 32) / K, 2);

    // Process each height row independently
    for (std::uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = !largest;  // Sort direction for bitonic sequence properties

        // Initial bitonic sort on local width chunk
        process_and_sort_tiles<network_stable, fused_keys, largest != 0, tie_order>(
            input_dfb_index,             // Input values buffer (double-buffered)
            index_dfb_index,             // Input indices buffer (double-buffered)
            input_transposed_dfb_index,  // Transposed values staging buffer
            index_transposed_dfb_index,  // Transposed indices staging buffer
            Wt,                          // Width tiles for this local chunk
            switch_dir,                  // Whether to alternate sort direction
            ascending,                   // Current sort direction
            end_phase);                  // Ending phase for local sort

        std::uint32_t num_k_sequences = (Wt * 32) / K;  // Number of K-element sequences in chunk

        // Iterative bitonic sort across the entire local width chunk
        // Perform log(Wt) iterations of divide-and-conquer merging:
        // - Iteration 0: Compare tiles (0,1), (2,3), (4,5), ... → pairs of 64 elements
        // - Iteration 1: Compare tiles (0,2), (4,6), (8,10), ... → groups of 128 elements
        // - Iteration n: Compare tiles with distance 2^n → groups of 64*(2^(n+1)) elements
        // Final iteration produces locally sorted TopK results for this width chunk.
        for (std::uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            process_iteration<network_stable, fused_keys, tie_order>(
                m_iter,                      // Current merge iteration (0 to logWt-1)
                K,                           // TopK value (number of elements to find)
                Wt,                          // Width tiles in local chunk
                num_k_sequences,             // Number of K-element sequences (updated each iter)
                tiles_per_seq,               // Tiles per sequence (ceil(K/32))
                input_transposed_dfb_index,  // Values buffer for in-place operations
                index_transposed_dfb_index,  // Indices buffer for in-place operations
                input_dest_start,            // Destination register 0 (first tile)
                input_dest_end,              // Destination register 1 (second tile)
                index_dest_start,            // Destination register 2 (first indices)
                index_dest_end,              // Destination register 3 (second indices)
                !direction_init,             // Base sort direction
                switch_dir,                  // Whether to switch direction per iteration
                logk,                        // log2(K) for bitonic network depth
                seq_per_2tiles,              // Sequences that fit in 2 tiles
                largest);                    // Find largest (true) or smallest (false)
        }  // m_iter loop

        // The first Kt tiles of the transposed buffers hold the local top k; repack them in the output formats.
        copy_front_tiles(input_transposed_dfb_index, values_dfb_index, Kt, Wt);
        if constexpr (!fused_keys) {
            // In fused mode the indices ride inside the packed value tiles; there is no index stream.
            copy_front_tiles(index_transposed_dfb_index, output_ind_dfb_index, Kt, Wt);
        }

        // Per round: [own Kt | partner Kt] from the landing CB, one merge step in place, survivors to the writer.
        for (std::uint32_t r = 0; r < num_recv_rounds; ++r) {
            copy_front_tiles(landing_values_dfb_index, merge_values_dfb_index, 2 * Kt, 2 * Kt);
            if constexpr (!fused_keys) {
                copy_front_tiles(landing_indices_dfb_index, merge_indices_dfb_index, 2 * Kt, 2 * Kt);
            }

            // Partners of the next round must be sorted in opposite directions, like direction_init does locally.
            const bool merge_ascending = (largest == 0) != (((core_id >> (r + 1)) & 1) == 1);
            std::uint32_t merge_num_k_sequences = (2 * Kt * 32) / K;
            uint32_t merge_seq_per_2tiles = std::max<uint32_t>((2 * 32) / K, 2);
            process_iteration<network_stable, fused_keys, tie_order>(
                0,                      // Single merge step: at m_iter 0 tile t is paired with tile t + Kt
                K,                      // TopK value
                2 * Kt,                 // Own Kt tiles followed by the partner's Kt tiles
                merge_num_k_sequences,  // Two K-element sequences
                tiles_per_seq,          // Tiles per sequence (ceil(K/32))
                merge_values_dfb_index,
                merge_indices_dfb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                !merge_ascending,  // process_iteration rebuilds the kept sequence with ascending = !largest
                switch_dir,
                logk,
                merge_seq_per_2tiles,
                largest);

            copy_front_tiles(merge_values_dfb_index, values_dfb_index, Kt, 2 * Kt);
            if constexpr (!fused_keys) {
                copy_front_tiles(merge_indices_dfb_index, output_ind_dfb_index, Kt, 2 * Kt);
            }
        }
    }  // ht loop
}
