// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "matmul_wo_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out = tt::CBIndex::c_2;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_3;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_out);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;
    const uint32_t num_tiles_h = matmul_wo_ring::K_TILES_PER_CORE_A[dram_bank_id];

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    constexpr uint32_t num_collectors = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint8_t collector_core_coords[num_collectors][2] = COLLECTOR_CORE_COORDS;

    // Get dst address
    const uint32_t local_collector_base_addr = get_write_ptr(cb_s2c_in2);
    uint64_t collector_dst_base_addr[num_collectors];
    for (uint32_t collector_idx = 0; collector_idx < num_collectors; ++collector_idx) {
        collector_dst_base_addr[collector_idx] = get_noc_addr(
            collector_core_coords[collector_idx][0],
            collector_core_coords[collector_idx][1],
            local_collector_base_addr);
    }

    constexpr uint32_t collector_dst_stride = 12 * out_tile_size;
    const uint32_t collector_offset = dram_bank_id * out_tile_size;
    uint32_t local_collector_dst_addr = local_collector_base_addr + collector_offset;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = matmul_wo_ring::W_TXNS_PER_BLOCK;
    constexpr uint32_t w_tiles_per_txn = matmul_wo_ring::W_TILES_PER_TXN;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t num_blocks_per_iter =
        (num_tiles_h * num_n_tiles_per_iter + w_tiles_per_block - 1) / w_tiles_per_block;
    const uint32_t w_total_blocks = num_blocks_per_iter * num_iters;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(reduce_semaphore_id);
    uint64_t partial_semaphore_noc_addr[num_collectors];
    for (uint32_t collector_idx = 0; collector_idx < num_collectors; ++collector_idx) {
        partial_semaphore_noc_addr[collector_idx] = get_noc_addr(
            collector_core_coords[collector_idx][0], collector_core_coords[collector_idx][1], semaphore_addr);
    }

    // Use semaphore_inc<posted=true> with 1 transaction ID and flush barrier for trids
    constexpr uint32_t semaphore_trid = 0x4;

    noc_async_write_set_trid(semaphore_trid, /*noc=*/1);

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_wait_front(cb_c2w_out, num_n_tiles_per_iter);
        uint32_t local_collector_src_addr = get_read_ptr(cb_c2w_out);

        for (uint32_t collector_idx = 0; collector_idx < num_collectors; ++collector_idx) {
            noc_async_write_one_packet_set_state</*posted=*/true>(
                collector_dst_base_addr[collector_idx], out_tile_size, /*noc=*/1, vchannel);
            noc_async_write_one_packet_with_state</*posted=*/true>(local_collector_src_addr, local_collector_dst_addr);
            noc_semaphore_inc</*posted=*/true>(
                partial_semaphore_noc_addr[collector_idx], /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

            // // Ensure write and semaphore have left the core before continuing
            // noc_async_write_flushed_with_trid(semaphore_trid, /*noc=*/1);

            local_collector_src_addr += out_tile_size;
        }

        cb_pop_front(cb_c2w_out, num_n_tiles_per_iter);

        local_collector_dst_addr += collector_dst_stride;
    }

    // Ensure write and semaphore have left the core before continuing
    noc_async_write_flushed_with_trid(semaphore_trid, /*noc=*/1);
}
