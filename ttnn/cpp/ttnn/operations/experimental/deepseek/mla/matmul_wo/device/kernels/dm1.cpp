// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "matmul_wo_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
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

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_out);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    const uint32_t num_tiles_h = matmul_wo_ring::K_TILES_PER_CORE_A[dram_bank_id];

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    // Get src address
    constexpr uint32_t collector_src_stride = num_n_tiles_per_iter * out_tile_size;
    uint32_t local_collector_src_addr = get_write_ptr(cb_c2w_out);

    // Get dst address
    constexpr uint32_t packet_size = num_n_tiles_per_iter * out_tile_size;
    const uint32_t local_collector_base_addr = get_write_ptr(cb_s2c_in);
    const uint64_t collector_dst_base_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, local_collector_base_addr);
    constexpr uint32_t collector_dst_stride = 12 * num_n_tiles_per_iter * out_tile_size;
    const uint32_t collector_offset = dram_bank_id * num_n_tiles_per_iter * out_tile_size;
    uint32_t local_collector_dst_addr = local_collector_base_addr + collector_offset;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = matmul_wo_ring::W_TXNS_PER_BLOCK;
    constexpr uint32_t w_tiles_per_txn = matmul_wo_ring::W_TILES_PER_TXN;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;
    const uint32_t num_blocks_per_iter =
        (num_tiles_h * num_n_tiles_per_iter + w_tiles_per_block - 1) / w_tiles_per_block;
    const uint32_t w_total_blocks = num_blocks_per_iter * num_iters;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(reduce_semaphore_id);
    const uint64_t partial_semaphore_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, semaphore_addr);

    // noc_async_write_one_packet_set_state</*posted=*/true>(collector_dst_base_addr, packet_size, /*noc=*/1, vchannel);

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_wait_front(cb_c2w_out, num_n_tiles_per_iter);

        // noc_async_write_one_packet_with_state</*posted=*/false>(local_collector_src_addr, local_collector_dst_addr);
        noc_semaphore_inc</*posted=*/true>(partial_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        cb_pop_front(cb_c2w_out, num_n_tiles_per_iter);

        // local_collector_src_addr += collector_src_stride;
        // local_collector_dst_addr += collector_dst_stride;
    }

    // Ensure write and semaphore have left the core before continuing
    noc_async_posted_atomic_barrier();
}
