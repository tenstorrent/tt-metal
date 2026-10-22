// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "topk_local_common.hpp"

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
    using namespace topk_local;

    std::uint32_t direction_init = get_arg_val<std::uint32_t>(0);

    init();

    uint32_t seq_per_2tiles = initial_seq_per_2tiles;

    // Process each height row independently
    for (std::uint32_t ht = 0; ht < Ht; ++ht) {
        sort_row(direction_init, seq_per_2tiles);

        // The writer sends these Kt tiles to the final aggregation core.
        pack_row_top_k();
    }
}
