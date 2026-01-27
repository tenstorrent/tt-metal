// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_core_id = get_named_compile_time_arg_val("reduce_core_id");
    constexpr uint32_t reduce_core_physical_x = get_named_compile_time_arg_val("reduce_core_physical_x");
    constexpr uint32_t reduce_core_physical_y = get_named_compile_time_arg_val("reduce_core_physical_y");

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
    const auto reduce_semaphore = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    constexpr uint32_t num_w_tiles_h = 20;
    constexpr uint32_t num_w_tiles_w = 8;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    constexpr uint32_t num_reduction_txns = num_w_tiles_w * out_tile_size / noc_packet_size;

    uint32_t semaphore_addr = get_semaphore(reduce_semaphore);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    const uint64_t reduce_semaphore_noc_addr =
        get_noc_addr(reduce_core_physical_x, reduce_core_physical_y, semaphore_addr);

    // Source and destination addresses for the all2all
    const uint32_t local_base_addr = get_write_ptr(cb_s2c_out);
    const uint64_t neighbor_base_addr = get_noc_addr(reduce_core_physical_x, reduce_core_physical_y, local_base_addr);

    // Precompute buffer offsets
    uint32_t local_addr = local_base_addr;
    uint32_t LOCAL_BUFFER_OFFSET[num_reduction_txns];
    for (uint32_t i = 0; i < num_reduction_txns; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_addr;
        local_addr += +noc_packet_size;
    }

    uint64_t REMOTE_BUFFER_OFFSET[num_reduction_txns];
    uint64_t remote_addr = neighbor_base_addr;
    for (uint32_t i = 0; i < num_reduction_txns; ++i) {
        REMOTE_BUFFER_OFFSET[i] = remote_addr;
        remote_addr += noc_packet_size;
    }

    // Set state for the writes
    noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_base_addr, noc_packet_size, /*noc=*/1, vchannel);

    //-------------------------------------------------------------------------

    cb_wait_front(cb_c2w_rdy, 1);
    cb_pop_front(cb_c2w_rdy, 1);

    if (dram_bank_id != reduce_core_id) {
        // Send the NOC transaction to the reduce core
        // We will need 2 transactions to send 8 tiles
        for (uint32_t txn_id = 0; txn_id < num_reduction_txns; ++txn_id) {
            noc_async_write_one_packet_with_state</*posted=*/true>(
                LOCAL_BUFFER_OFFSET[txn_id], REMOTE_BUFFER_OFFSET[txn_id]);
        }

        // Signal the semaphore to the reduce core
        noc_semaphore_inc</*posted=*/true>(reduce_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        noc_async_posted_atomic_barrier();
    }

    if (dram_bank_id == reduce_core_id) {
        // If we are the reduce core, we need to wait for the data to be ready
        noc_semaphore_wait_min(my_semaphore_ptr, num_cores - 1);

        // TODO: Reduce the data using compute core. Skip for now.
    }
}
