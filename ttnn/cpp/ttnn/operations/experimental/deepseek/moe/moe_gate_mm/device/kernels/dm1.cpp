// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ckernel_defs.h"
#include "api/tile_constants.h"
#include "api/debug/dprint_pages.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t num_mask_receivers = get_named_compile_time_arg_val("num_mask_receivers");

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
    const auto collector_semaphore = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w_id = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in_id = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy_id = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2_id = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out_id = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3_id = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4_id = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5_id = tt::CBIndex::c_7;
    constexpr auto cb_w2c_in6_id = tt::CBIndex::c_8;
    constexpr auto cb_w2c_in7_id = tt::CBIndex::c_9;

    // Aliases
    constexpr auto cb_w2c_in8_id = tt::CBIndex::c_6;

    CircularBuffer cb_c2w_rdy(cb_c2w_rdy_id);
    CircularBuffer cb_w2c_in2(cb_w2c_in2_id);
    CircularBuffer cb_s2c_out(cb_s2c_out_id);
    CircularBuffer cb_w2c_in3(cb_w2c_in3_id);
    CircularBuffer cb_w2c_in4(cb_w2c_in4_id);
    CircularBuffer cb_w2c_in5(cb_w2c_in5_id);
    CircularBuffer cb_w2c_in6(cb_w2c_in6_id);
    CircularBuffer cb_w2c_in7(cb_w2c_in7_id);
    CircularBuffer cb_w2c_in8(cb_w2c_in8_id);

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in_id);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w_id);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out_id);

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = is_send_core ? 2 * 72 : 2 * 76;
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    // This kernel issues everything on NoC 1 (the writer NoC on Wormhole).
    Noc noc_obj(1);
    Semaphore<> partial_sem(partial_semaphore);
    // The raw address is still needed for the remote partial-ready signal below.
    const uint32_t semaphore_addr = get_semaphore(partial_semaphore);

    const uint64_t partial_semaphore_noc_addr1 =
        get_noc_addr(neighbor1_physical_x, neighbor1_physical_y, semaphore_addr);

    const uint64_t partial_semaphore_noc_addr2 =
        get_noc_addr(neighbor2_physical_x, neighbor2_physical_y, semaphore_addr);

    const uint32_t local_src_addr = cb_s2c_out.get_write_ptr();
    const uint32_t local_dst_addr = cb_w2c_in2.get_write_ptr();

    // Use semaphore_inc<posted=true> with 1 transaction ID and flush barrier for trids
    constexpr uint32_t semaphore_trid = 0x4;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    // Last non-sender tile id; it equals the group-mask receiver count.
    constexpr uint32_t COLLECTOR_CORE_ID = num_mask_receivers;

    const uint32_t local_collector_addr = 1024 + cb_w2c_in4.get_write_ptr();
    const uint32_t collector_offset = core_id * out_tile_size;

    constexpr uint32_t group_score_size = tt::constants::FACE_WIDTH * sizeof(uint16_t);

    //-------------------------------------------------------------------------
    // Group scores (7 cores -> 1 collector core)
    //-------------------------------------------------------------------------

    // Group scores: data exists in 1st and 4th row of the first face of the bfloat16 tile
    constexpr uint32_t group_score_offset1 = 0;
    constexpr uint32_t group_score_offset2 = 4 * group_score_size;
    const uint32_t local_group_score_base_addr = cb_c2w_rdy.get_write_ptr();
    const uint32_t local_group_score_src_addr1 = local_group_score_base_addr + group_score_offset1;
    const uint32_t local_group_score_src_addr2 = local_group_score_base_addr + group_score_offset2;

    const uint32_t local_group_scores_dst_addr1 = local_collector_addr + core_id * group_score_size;
    const uint32_t local_group_scores_dst_addr2 = local_group_scores_dst_addr1 + 8 * group_score_size;

    // The collector counts group-score and top8 arrivals on its own semaphore. Sharing the partial
    // semaphore let an early increment satisfy the collector's partial wait and get zeroed, and the
    // neighbor's partial write (a set to 1) overwrite increments that had already landed.
    Semaphore<> collector_sem(collector_semaphore);
    const uint32_t collector_semaphore_addr = get_semaphore(collector_semaphore);
    const uint64_t collector_semaphore_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, collector_semaphore_addr);

    //-------------------------------------------------------------------------
    // Group masks (1 collector core -> 7 cores)
    //-------------------------------------------------------------------------
    const uint32_t local_group_masks_addr = cb_w2c_in5.get_write_ptr();
    //-------------------------------------------------------------------------
    // Top8 partials (7 cores -> 1 collector core)
    //-------------------------------------------------------------------------
    constexpr uint32_t partials_size = 2 * tt::constants::FACE_HW * sizeof(uint16_t);  // values and indices

    // Top8 partials: data exists in first 16 rows
    const uint32_t local_top8_partials_src_addr = cb_w2c_in8.get_write_ptr();

    const uint32_t local_top8_dst_base_addr = cb_w2c_in6.get_write_ptr();
    const uint32_t local_top8_dst_addr = local_top8_dst_base_addr + core_id * partials_size;

    //-------------------------------------------------------------------------
    // Semaphores start at 0 on every launch (host-initialized). Do not zero them here: a neighbor's
    // partial write can land before this core gets this far.
    if (is_send_core) {
        // Since neighbor2 is farther, we send it first.
        // Set state for the writes
        noc_obj.set_async_write_state<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            out_tile_size,
            {.noc_x = neighbor2_physical_x, .noc_y = neighbor2_physical_y, .addr = local_dst_addr},
            {.vc = vchannel});

        // Wait for the data1 to be ready
        cb_c2w_rdy.wait_front(1);

        // Send the data to the neighbor1
        noc_obj.async_write_with_state<NocOptions::POSTED, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(local_src_addr),
            UnicastEndpoint{},
            out_tile_size,
            {},
            {.noc_x = neighbor2_physical_x, .noc_y = neighbor2_physical_y, .addr = local_dst_addr});

        // Signal neighbor2 that data is ready (increment their semaphore)
        noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
            partial_semaphore_noc_addr2, /*val=*/1, /*be=*/0xF, /*cmd_buf=*/write_at_cmd_buf, /*noc=*/1, vchannel);
        noc_inline_dw_write_with_state<
            /*update_addr_lo=*/false,
            /*update_counter=*/true,
            /*posted=*/true,
            /*update_addr_hi=*/false,
            /*update_val=*/false>(/*val=*/1);

        // Ensure write and semaphore have left the core before continuing
        noc_obj.async_writes_flushed<NocOptions::POSTED>();

        cb_c2w_rdy.pop_front(1);

        noc_obj.set_async_write_state<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            out_tile_size,
            {.noc_x = neighbor1_physical_x, .noc_y = neighbor1_physical_y, .addr = local_dst_addr},
            {.vc = vchannel});

        // Wait for the data2 to be ready
        cb_c2w_rdy.wait_front(1);

        // Send the data to the neighbor1
        noc_obj.async_write_with_state<NocOptions::POSTED, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(local_src_addr),
            UnicastEndpoint{},
            out_tile_size,
            {},
            {.noc_x = neighbor1_physical_x, .noc_y = neighbor1_physical_y, .addr = local_dst_addr});

        // Signal neighbor1 that data is ready (increment their semaphore)
        noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
            partial_semaphore_noc_addr1, /*val=*/1, /*be=*/0xF, /*cmd_buf=*/write_at_cmd_buf, /*noc=*/1, vchannel);
        noc_inline_dw_write_with_state<
            /*update_addr_lo=*/false,
            /*update_counter=*/true,
            /*posted=*/true,
            /*update_addr_hi=*/false,
            /*update_val=*/false>(/*val=*/1);

        // Ensure write and semaphore have left the core before continuing
        noc_obj.async_writes_flushed<NocOptions::POSTED>();

        cb_c2w_rdy.pop_front(1);

        return;
    }

    // -------------------------------------------------------------------------
    // Rest of the 8 cores do more
    // -------------------------------------------------------------------------
    cb_w2c_in2.reserve_back(1);

    // Wait for the data from sender cores to be ready
    partial_sem.wait_min(1);
    partial_sem.set(0);
    cb_w2c_in2.push_back(1);

    // Wait for group scores to be ready from compute
    cb_c2w_rdy.wait_front(1);

    //-------------------------------------------------------------------------
    // Cores sending data to the collector core
    //-------------------------------------------------------------------------
    if (core_id != COLLECTOR_CORE_ID) {
        noc_obj.set_async_write_state<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            group_score_size,
            {.noc_x = collector_physical_x, .noc_y = collector_physical_y, .addr = local_collector_addr},
            {.vc = vchannel});

        // Send them over to the collector core
        noc_obj.async_write_with_state<NocOptions::POSTED, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(local_group_score_src_addr1),
            UnicastEndpoint{},
            group_score_size,
            {},
            {.noc_x = collector_physical_x, .noc_y = collector_physical_y, .addr = local_group_scores_dst_addr1});
        noc_obj.async_write_with_state<NocOptions::POSTED, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(local_group_score_src_addr2),
            UnicastEndpoint{},
            group_score_size,
            {},
            {.noc_x = collector_physical_x, .noc_y = collector_physical_y, .addr = local_group_scores_dst_addr2});

        // Signal the collector core that data is ready (increment their semaphore)
        noc_async_write_set_trid(semaphore_trid, /*noc=*/1);
        noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);
        noc_async_write_flushed_with_trid(semaphore_trid, /*noc=*/1);
        noc_async_write_set_trid(0, /*noc=*/1);

        // We are done with the group scores
        cb_c2w_rdy.pop_front(1);

        cb_w2c_in5.reserve_back(1);

        // Wait for the group mask to be ready in the collector core
        partial_sem.wait_min(1);
        partial_sem.set(0);

        // Let compute know we got the group masks
        cb_w2c_in5.push_back(1);

        // Wait for compute to send the top-8 values and indices
        cb_w2c_in8.wait_front(1);

        noc_obj.set_async_write_state<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            partials_size,
            {.noc_x = collector_physical_x, .noc_y = collector_physical_y, .addr = local_collector_addr},
            {.vc = vchannel});

        // Send them over to the collector core
        noc_obj.async_write_with_state<NocOptions::POSTED, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(local_top8_partials_src_addr),
            UnicastEndpoint{},
            partials_size,
            {},
            {.noc_x = collector_physical_x, .noc_y = collector_physical_y, .addr = local_top8_dst_addr});

        // Signal the collector core that data is ready (increment their semaphore)
        noc_async_write_set_trid(semaphore_trid, /*noc=*/1);
        noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);
        noc_async_write_flushed_with_trid(semaphore_trid, /*noc=*/1);
        noc_async_write_set_trid(0, /*noc=*/1);

        cb_w2c_in8.pop_front(1);
    }

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    if (core_id == COLLECTOR_CORE_ID) {
        // Rejig the group scores to be in the correct location -> where everyone else also puts it
        CoreLocalMem<volatile uint32_t> src_ptr(local_group_score_src_addr1);
        CoreLocalMem<volatile uint32_t> dst_ptr(local_group_scores_dst_addr1);
        for (uint32_t i = 0; i < 8; i++) {
            dst_ptr[i] = src_ptr[i];
        }

        // Same, but for the second row of the group scores
        src_ptr = CoreLocalMem<volatile uint32_t>(local_group_score_src_addr2);
        dst_ptr = CoreLocalMem<volatile uint32_t>(local_group_scores_dst_addr2);
        for (uint32_t i = 0; i < 8; i++) {
            dst_ptr[i] = src_ptr[i];
        }

        // We are done with the group scores
        cb_c2w_rdy.pop_front(1);

        cb_w2c_in4.reserve_back(1);

        // I am collecting, let us wait for everyone else to finish sending their data to me
        collector_sem.wait_min(num_mask_receivers);

        // Let compute know that we got the group scores
        cb_w2c_in4.push_back(1);

        //-----------------------------------------------------------------
        // Get the group masks
        //-----------------------------------------------------------------
        cb_w2c_in5.wait_front(1);
        // Unicast each worker. The ring rectangle includes a dispatch core when dispatch is on a row.
        for (uint32_t receiver = 0; receiver < num_mask_receivers; ++receiver) {
            const uint32_t receiver_x = get_arg_val<uint32_t>(argidx + (receiver * 2));
            const uint32_t receiver_y = get_arg_val<uint32_t>(argidx + (receiver * 2) + 1);
            noc_obj.async_write<NocOptions::CUSTOM_VC>(
                CoreLocalMem<uint32_t>(local_group_masks_addr),
                UnicastEndpoint{},
                /*size_bytes=*/2048,
                {},
                {.noc_x = receiver_x, .noc_y = receiver_y, .addr = local_group_masks_addr},
                {.vc = vchannel});
        }
        noc_obj.async_write_barrier();
        for (uint32_t receiver = 0; receiver < num_mask_receivers; ++receiver) {
            const uint32_t receiver_x = get_arg_val<uint32_t>(argidx + (receiver * 2));
            const uint32_t receiver_y = get_arg_val<uint32_t>(argidx + (receiver * 2) + 1);
            partial_sem.up(noc_obj, receiver_x, receiver_y, 1);
        }
        noc_obj.async_atomic_barrier();

        cb_w2c_in5.pop_front(1);

        cb_w2c_in6.reserve_back(4);

        // Wait for the top8 partials to arrive (7 group-score + 7 top8 increments)
        collector_sem.wait_min(num_mask_receivers + num_mask_receivers);

        // Let compute know that we got the top8 partials
        cb_w2c_in6.push_back(4);

        // Wait for final top8 to be ready from compute
        cb_c2w_rdy.wait_front(1);
        cb_c2w_rdy.pop_front(1);
    }
}
