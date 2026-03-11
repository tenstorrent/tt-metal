// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/api/tt-metalium/constants.hpp"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_semaphore_id = get_named_compile_time_arg_val("collector_semaphore_id");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t first_physical_x = get_named_compile_time_arg_val("first_physical_x");
    constexpr uint32_t first_physical_y = get_named_compile_time_arg_val("first_physical_y");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto rope_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<rope_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto is_collector = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto rope_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto pos = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_r2c_rope = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_c2w_x2 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_x2 = tt::CBIndex::c_6;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t rope_tile_size = get_tile_size(cb_r2c_rope);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);
    constexpr uint32_t partial_x2_sum_tile_size = get_tile_size(cb_c2w_x2);

    // Read variables
    const uint32_t local_x2_src_addr = get_read_ptr(cb_c2w_x2);

    // Collector variables
    constexpr uint32_t local_collector_pkt_size = 8 * tt::constants::FACE_WIDTH * sizeof(uint16_t);  // We move 8 rows
    uint32_t local_collector_base_addr = get_write_ptr(cb_w2c_x2);
    const uint32_t local_collector_offset = dram_bank_id * local_collector_pkt_size;
    const uint32_t local_collector_dst_addr = local_collector_base_addr + local_collector_offset;
    const uint64_t collector_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, local_collector_dst_addr);

    uint32_t semaphore_addr = get_semaphore(collector_semaphore_id);
    const uint64_t collector_semaphore_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, semaphore_addr);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    *my_semaphore_ptr = 0;

    //-------------------------------------------------------------------------
    // RMSNorm scale factor (1 collector core -> 11 cores)
    //-------------------------------------------------------------------------
    const uint32_t local_rms_scale_addr = get_write_ptr(cb_w2c_x2);
    const uint64_t rms_scale_noc_addr = get_noc_multicast_addr(
        first_physical_x, first_physical_y, collector_physical_x, collector_physical_y, local_rms_scale_addr);
    const uint64_t rms_scale_semaphore_noc_addr = get_noc_multicast_addr(
        first_physical_x, first_physical_y, collector_physical_x, collector_physical_y, semaphore_addr);

    // Use semaphore_inc<posted=true> with 1 transaction ID and flush barrier for trids
    constexpr uint32_t semaphore_trid = 0x4;

    //-------------------------------------------------------------------------
    // RMSNorm
    //-------------------------------------------------------------------------
    // Wait for the partial X^2 sum to arrive from compute.
    cb_wait_front(cb_c2w_x2, 2);

    noc_async_write_set_trid(semaphore_trid, /*noc=*/1);

    noc_async_write_one_packet_set_state</*posted=*/true>(
        collector_noc_addr, local_collector_pkt_size, /*noc=*/1, vchannel);
    noc_async_write_one_packet_with_state</*posted=*/true>(local_x2_src_addr, local_collector_dst_addr);
    noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);
    noc_async_write_flushed_with_trid(semaphore_trid, /*noc=*/1);

    cb_pop_front(cb_c2w_x2, 2);

    //-------------------------------------------------------------------------
    if (is_collector) {
        cb_reserve_back(cb_w2c_x2, 2);
        noc_semaphore_wait_min(my_semaphore_ptr, num_cores);
        *my_semaphore_ptr = 0;
        cb_push_back(cb_w2c_x2, 2);

        cb_wait_front(cb_c2w_x2, 2);
        // Multicast this data to all the cores
        noc_async_write_multicast_one_packet(local_x2_src_addr, rms_scale_noc_addr, /*size=*/4096, /*num_dests=*/11);

        // Set the semaphore to let the clients know they got the data
        *my_semaphore_ptr = 1;
        noc_semaphore_set_multicast(
            semaphore_addr, rms_scale_semaphore_noc_addr, /*num_dests=*/11, /*linked=*/false, /*noc=*/1);

        // Loop back the data to compute
        cb_reserve_back(cb_w2c_x2, 2);
        noc_async_write_one_packet_set_state</*posted=*/false>(collector_noc_addr, 4096, /*noc=*/1, vchannel);
        noc_async_write_one_packet_with_state</*posted=*/false>(local_x2_src_addr, get_write_ptr(cb_w2c_x2));
        cb_push_back(cb_w2c_x2, 2);

        cb_pop_front(cb_c2w_x2, 2);
    } else {
        // Wait for the RMSNorm scale factor to arrive from collector
        cb_reserve_back(cb_w2c_x2, 2);
        noc_semaphore_wait_min(my_semaphore_ptr, 1);
        *my_semaphore_ptr = 0;
        cb_push_back(cb_w2c_x2, 2);
    }
}
