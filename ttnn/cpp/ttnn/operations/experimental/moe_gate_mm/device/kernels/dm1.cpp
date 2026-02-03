// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel_defs.h"
#include "tt-metalium/constants.hpp"
#include "api/debug/dprint_pages.h"

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
    const auto is_send_core = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto collector_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto collector_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5 = tt::CBIndex::c_7;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = is_send_core ? 2 * 72 : 2 * 76;
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
    // Collector core
    //-------------------------------------------------------------------------
    const uint32_t local_collector_addr = get_write_ptr(cb_w2c_in3);
    const uint64_t collector_dst_base_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, local_collector_addr);
    const uint32_t collector_offset = core_id * out_tile_size;

    constexpr uint32_t group_score_size = tt::constants::FACE_WIDTH * sizeof(uint16_t);

    // Figure out where to read and write the adjusted scores
    const uint32_t local_adj_scores_src_addr = get_write_ptr(cb_s2c_out);
    const uint32_t local_adj_scores_dst_base_addr = get_write_ptr(cb_w2c_in3);
    const uint32_t local_adj_scores_dst_addr = local_adj_scores_dst_base_addr + collector_offset;

    // Find out where to read and write the raw scores
    const uint32_t local_raw_scores_src_addr = get_write_ptr(cb_w2c_in5);
    const uint32_t local_raw_scores_dst_base_addr = get_write_ptr(cb_w2c_in4);
    const uint32_t local_raw_scores_dst_addr = local_raw_scores_dst_base_addr + collector_offset;
    const uint32_t local_group_scores_dst_addr1 =
        local_raw_scores_dst_base_addr + 7 * out_tile_size + core_id * group_score_size;
    const uint32_t local_group_scores_dst_addr2 = local_group_scores_dst_addr1 + 8 * group_score_size;

    // Group scores: data exists in 1st and 4th row of the first face of the bfloat16 tile
    constexpr uint32_t group_score_offset1 = 0;
    constexpr uint32_t group_score_offset2 = 4 * group_score_size;
    const uint32_t local_group_score_base_addr = get_write_ptr(cb_c2w_rdy);
    const uint32_t local_group_score_src_addr1 = local_group_score_base_addr + group_score_offset1;
    const uint32_t local_group_score_src_addr2 = local_group_score_base_addr + group_score_offset2;

    const uint64_t collector_semaphore_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, semaphore_addr);

    //-------------------------------------------------------------------------

    if (is_send_core) {
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

    if (!is_send_core) {
        cb_reserve_back(cb_w2c_in2, 1);
        // Wait for the data to be ready
        noc_semaphore_wait_min(my_semaphore_ptr, 1);
        cb_push_back(cb_w2c_in2, 1);

        // Wait for original and bias adjusted scores to be ready
        cb_wait_front(cb_c2w_rdy, 1);

        //-------------------------------------------------------------------------
        // Cores sending data to the collector core
        //-------------------------------------------------------------------------
        if (core_id != 7) {
            // Send them over to the core collecting them
            noc_async_write_one_packet_set_state</*posted=*/true>(
                collector_dst_base_addr, out_tile_size, /*noc=*/1, vchannel);

            // First send the bias adjusted scores
            noc_async_write_one_packet_with_state</*posted=*/true>(
                local_adj_scores_src_addr, local_adj_scores_dst_addr);

            // Then send the raw scores
            noc_async_write_one_packet_with_state</*posted=*/true>(
                local_raw_scores_src_addr, local_raw_scores_dst_addr);

            // Wait for the writes to be flushed
            noc_async_posted_writes_flushed();
            cb_pop_front(cb_c2w_rdy, 1);

            noc_async_write_one_packet_set_state</*posted=*/true>(
                collector_dst_base_addr, group_score_size, /*noc=*/1, vchannel);

            // Wait for the column scores to be ready
            cb_wait_front(cb_c2w_rdy, 1);

            // Send them over to the collector core
            noc_async_write_one_packet_with_state</*posted=*/true>(
                local_group_score_src_addr1, local_group_scores_dst_addr1);
            noc_async_write_one_packet_with_state</*posted=*/true>(
                local_group_score_src_addr2, local_group_scores_dst_addr2);

            // Signal neighbor2 that data is ready (increment their semaphore)
            noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

            // Ensure write and semaphore have left the core before continuing
            noc_async_posted_atomic_barrier();

            cb_pop_front(cb_c2w_rdy, 1);
        }

        //-------------------------------------------------------------------------
        // Collector core
        //-------------------------------------------------------------------------
        if (core_id == 7) {
            cb_pop_front(cb_c2w_rdy, 1);

            // Wait for the column scores to be ready
            cb_wait_front(cb_c2w_rdy, 1);

            // Rejig the group scores to be in the correct location -> where everyone else also puts it
            auto src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_score_src_addr1);
            auto dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_scores_dst_addr1);
            for (uint32_t i = 0; i < 8; i++) {
                dst_ptr[i] = src_ptr[i];
            }

            // Same, but for the second row of the group scores
            src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_score_src_addr2);
            dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_scores_dst_addr2);
            for (uint32_t i = 0; i < 8; i++) {
                dst_ptr[i] = src_ptr[i];
            }

            // We are done with the group scores
            cb_pop_front(cb_c2w_rdy, 1);

            cb_reserve_back(cb_w2c_in3, 7 + 1);

            // I am collecting, let us wait for everyone else to finish sending their data to me
            noc_semaphore_wait_min(my_semaphore_ptr, 1 + 7);

            // Let compute know that we got bias adjusted scores and the group scores
            // 7 bias adjusted scores from other cores + 1 tile of group scores
            // Our own bias adjusted scores are already in cb_s2c_out
            cb_push_back(cb_w2c_in3, 7 + 1);

            //-----------------------------------------------------------------
            // Raw scores
            //-----------------------------------------------------------------
            cb_reserve_back(cb_w2c_in4, 7);

            // Let compute know we got the raw scores
            // 7 raw scores from other cores
            // Our own raw scores are already in cb_w2c_in5
            cb_push_back(cb_w2c_in4, 7);
        }
    }
}
