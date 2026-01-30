// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

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
    const auto partial_semaphore = get_arg_val<uint32_t>(argidx++);
    const auto send_core = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = send_core ? 2 * 72 : 2 * 76;
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(partial_semaphore);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    const uint64_t partial_semaphore_noc_addr1 =
        get_noc_addr(neighbor1_physical_x, neighbor1_physical_y, semaphore_addr);

    const uint64_t partial_semaphore_noc_addr2 =
        get_noc_addr(neighbor2_physical_x, neighbor2_physical_y, semaphore_addr);

    const uint32_t local_src_addr = get_write_ptr(cb_s2c_out);
    const uint32_t local_dst_addr = get_write_ptr(cb_w2c_in2);
    const uint64_t neighbor_dst_addr1 = get_noc_addr(neighbor1_physical_x, neighbor1_physical_y, local_dst_addr);
    const uint64_t neighbor_dst_addr2 = get_noc_addr(neighbor2_physical_x, neighbor2_physical_y, local_dst_addr);

    //-------------------------------------------------------------------------

    if (send_core) {
        // Since neighbor2 is farther, we send it first.
        // Set state for the writes
        noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_dst_addr2, out_tile_size, /*noc=*/1, vchannel);

        // Wait for the data1 to be ready
        cb_wait_front(cb_c2w_rdy, 1);

        // Send the data to the neighbor1
        noc_async_write_one_packet_with_state</*posted=*/true>(local_src_addr, neighbor_dst_addr2);

        // Signal neighbor1 that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(partial_semaphore_noc_addr2, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        cb_pop_front(cb_c2w_rdy, 1);

        noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_dst_addr1, out_tile_size, /*noc=*/1, vchannel);

        // Wait for the data2 to be ready
        cb_wait_front(cb_c2w_rdy, 1);

        // Send the data to the neighbor2
        noc_async_write_one_packet_with_state</*posted=*/true>(local_src_addr, neighbor_dst_addr1);

        // Signal neighbor2 that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(partial_semaphore_noc_addr1, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        cb_pop_front(cb_c2w_rdy, 1);
    }

    if (!send_core) {
        cb_reserve_back(cb_w2c_in2, 1);
        // Wait for the data to be ready
        noc_semaphore_wait_min(my_semaphore_ptr, 1);
        cb_push_back(cb_w2c_in2, 1);

        // Wait for the final data to be ready
        cb_wait_front(cb_c2w_rdy, 1);
        cb_pop_front(cb_c2w_rdy, 1);
    }
}
