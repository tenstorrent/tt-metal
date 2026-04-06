// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

#include "topk_common_funcs.hpp"

#include <cstdint>

/**
 * TopK Multicore Compute Kernel Implementation - Final Aggregation Phase
 *
 * This kernel implements the final aggregation stage of the multicore TopK algorithm.
 * It receives locally optimal TopK results from multiple worker cores and performs
 * a final bitonic merge to produce globally optimal TopK values and indices.
 *
 * ================================================================================================
 * FINAL AGGREGATION PHASE - GLOBAL TopK COMPUTATION
 * ================================================================================================
 *
 * ALGORITHM OVERVIEW:
 * 1. Receive Kt tiles from each of the (num_cores-1) local processing cores
 * 2. Treat received data as Wt_final width of tiles containing candidate TopK elements
 * 3. Apply the same bitonic merge algorithm as local cores to find global optimum
 * 4. Output final TopK values and indices in the desired format
 *
 * INPUT DATA STRUCTURE:
 * - input_cb_index/index_cb_index: Received data from all local cores
 * - Data layout: [Core0_TopK][Core1_TopK]...[CoreN_TopK] each of size Kt tiles
 * - Total width: Wt_final = num_local_cores * Kt tiles
 *
 * PROCESSING DIFFERENCES FROM LOCAL CORES:
 * - No initial local sort needed (data already sorted per core)
 * - Copy input data to transposed buffers for in-place bitonic operations
 * - Apply log(Wt_final) bitonic merge iterations
 * - Final output is the globally optimal TopK result
 */

void kernel_main() {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);  // Leftover row size after multicore processing
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t Kt = get_compile_time_arg_val(9);
    constexpr uint32_t logk = get_compile_time_arg_val(10);
    constexpr uint32_t logWt = get_compile_time_arg_val(11);
    constexpr uint32_t largest = get_compile_time_arg_val(12);
    constexpr uint32_t sorted = get_compile_time_arg_val(13);

    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    constexpr uint32_t tiles_per_seq = (K + 31) / 32;
    bool switch_dir = (K == 64);
    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);

    // init pack, compute and unpack
    init_sfpu(input_cb_index, values_cb_index);
    ckernel::topk_tile_init();

    // Aggregate results from all local cores for each height row
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        cb_wait_front(input_cb_index, Wt);  // Wait for all local TopK results (values)
        cb_wait_front(index_cb_index, Wt);  // Wait for all local TopK results (indices)

        // Use separate buffers to avoid racing conditions with reader kernel.
        // The reader kernel manages input_cb_index/index_cb_index, while compute
        // operations require separate staging buffers for in-place bitonic operations.

        pack_reconfig_data_format(input_transposed_cb_index);
        // Copy all received value tiles from local cores to transposed staging buffer
        for (uint32_t wt = 0; wt < Wt; wt++) {
            acquire_dst();
            cb_reserve_back(input_transposed_cb_index, 1);
            copy_tile(input_cb_index, wt, 0);         // Copy tile from local core wt
            pack_tile(0, input_transposed_cb_index);  // Pack to staging buffer
            release_dst();
        }  // wt loop
        cb_push_back(input_transposed_cb_index, Wt);
        cb_wait_front(input_transposed_cb_index, Wt);
        cb_pop_front(input_cb_index, Wt);  // Release input buffer space

        // Copy all received index tiles from local cores to transposed staging buffer
        copy_tile_to_dst_init_short_with_dt(input_cb_index, index_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            acquire_dst();
            cb_reserve_back(index_transposed_cb_index, 1);
            copy_tile(index_cb_index, wt, 0);         // Copy index tile from local core wt
            pack_tile(0, index_transposed_cb_index);  // Pack to staging buffer
            cb_push_back(index_transposed_cb_index, 1);
            release_dst();
        }  // wt loop
        cb_wait_front(index_transposed_cb_index, Wt);
        cb_pop_front(index_cb_index, Wt);  // Release input buffer space

        uint32_t num_k_sequences = (Wt * 32) / K;  // K-element sequences across all local results

        // Bitonic merge iterations to compute global TopK
        // Apply the same log(Wt_final) bitonic merge iterations as local cores,
        // but now operating on the aggregated results from all cores.
        // This produces the globally optimal TopK from all local TopK results.
        //
        // Merge pattern for Wt_final tiles:
        // - Iteration 0: Merge (0,1), (2,3), (4,5), ... from different local cores
        // - Iteration 1: Merge (0,2), (4,6), (8,10), ... across core boundaries
        // - Final iteration: Global TopK across all cores' contributions
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            process_iteration(
                m_iter,                     // Current merge iteration
                K,                          // TopK value
                Wt,                         // Total width tiles (from all cores)
                num_k_sequences,            // K-sequences in aggregated data
                tiles_per_seq,              // Tiles per sequence (ceil(K/32))
                input_transposed_cb_index,  // Aggregated values buffer
                index_transposed_cb_index,  // Aggregated indices buffer
                input_dest_start,           // Destination register 0
                input_dest_end,             // Destination register 1
                index_dest_start,           // Destination register 2
                index_dest_end,             // Destination register 3
                largest,                    // Sort direction
                switch_dir,                 // Direction switching strategy
                logk,                       // log2(K) for bitonic depth
                seq_per_2tiles,             // Sequences per tile pair
                largest);                   // Find largest vs smallest
        }

        // Extract the globally optimal TopK values and indices and prepare
        // for final output. Transpose back to WH format as required.

        // Extract and output final TopK values (first Kt tiles contain global optimum)
        transpose_and_pack(input_transposed_cb_index, values_cb_index, Kt, Wt);

        // Extract and output final TopK indices (corresponding to global optimum values)
        transpose_and_pack(index_transposed_cb_index, output_ind_cb_index, Kt, Wt);
    }  // ht loop
}
