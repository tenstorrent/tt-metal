// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Triple buffering constants
#define NUM_SLOTS 3  // 3 slots in CB

// Helper macros for counter advancement (avoids modulo on RISC-V)
#define ADVANCE_SLOT(s)       \
    do {                      \
        (s)++;                \
        if ((s) >= NUM_SLOTS) \
            (s) = 0;          \
    } while (0)
#define ADVANCE_TRID(t)      \
    do {                     \
        (t)++;               \
        if ((t) > NUM_SLOTS) \
            (t) = 1;         \
    } while (0)

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t first_physical_x = get_named_compile_time_arg_val("first_physical_x");
    constexpr uint32_t first_physical_y = get_named_compile_time_arg_val("first_physical_y");

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
    const auto raw_scores_semaphore = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5 = tt::CBIndex::c_7;
    constexpr auto cb_w2c_in6 = tt::CBIndex::c_8;
    constexpr auto cb_w2c_in7 = tt::CBIndex::c_9;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = is_send_core ? (2 * 72) : (2 * 76 + 1);
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 8;
    constexpr uint32_t w_tiles_per_txn = noc_packet_size / 2048;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t w_num_blocks = num_w_tiles_h * num_w_tiles_w / w_tiles_per_block;
    const uint32_t w_remaining_tiles = (num_w_tiles_h * num_w_tiles_w) % w_tiles_per_block;
    const uint32_t w_last_block_txns = (w_remaining_tiles + w_tiles_per_txn - 1) / w_tiles_per_txn;  // Ceiling division

    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_bytes_per_block = w_tiles_per_block * w_tile_size;
    constexpr uint32_t w_bytes_per_txn = w_tiles_per_txn * w_tile_size;

    // DRAM bank's base NOC address
    const uint64_t dram_noc_addr =
        get_noc_addr_from_bank_id</*DRAM=*/true>(dram_bank_id, /*bank_address_offset=*/w_addr);

    //-------------------------------------------------------------------------
    // CB addresses
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = get_write_ptr(cb_r2c_w);

    // Precompute slot addresses (avoid multiply in hot loop)
    // Each slot holds txns_per_block tiles
    uint32_t slot_addr[NUM_SLOTS][w_txns_per_block];

    uint32_t slot_addr_offset = 0;
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        for (uint32_t j = 0; j < w_txns_per_block; ++j) {
            slot_addr[i][j] = w_cb_base_addr + slot_addr_offset;
            slot_addr_offset += w_bytes_per_txn;
        }
    }

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    // Set w0_w1 state once before loop (will be reused for all experts)
    noc_async_read_one_packet_set_state<true>(dram_noc_addr, w_bytes_per_txn, vchannel);

    //-------------------------------------------------------------------------
    // Variables to track pipeline state
    //-------------------------------------------------------------------------
    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    bool txns_in_flight = false;

    // We reserve one to kick start the pipeline, and then it is steady state
    cb_reserve_back(cb_r2c_w, w_tiles_per_block);

    //-------------------------------------------------------------------------
    // Pipelined reading of W0/W1
    //-------------------------------------------------------------------------
    uint32_t w_dram_read_offset = 0;

    for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
        // Issue reads with current trid
        noc_async_read_set_trid(trid_to_issue);
        for (uint32_t txn_id = 0; txn_id < w_txns_per_block; ++txn_id) {
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                dram_noc_addr, w_dram_read_offset, slot_addr[slot_to_issue][txn_id], trid_to_issue);
            w_dram_read_offset += w_bytes_per_txn;
        }

        ADVANCE_SLOT(slot_to_issue);
        ADVANCE_TRID(trid_to_issue);

        // Only when we first start the pipeline, we don't have any txns in flight
        if (txns_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_r2c_w, w_tiles_per_block);

            ADVANCE_TRID(trid_to_wait);

            // Reserve for next block
            // Reserve back is not incremental, so to reserve one more, we need to reserve 2
            // This accounts for the one we already have reserved (for in-flight read)
            cb_reserve_back(cb_r2c_w, w_tiles_per_block * 2);
        }
        txns_in_flight = true;
    }

    // Issue txns for the last block
    noc_async_read_set_trid(trid_to_issue);
    for (uint32_t txn_id = 0; txn_id < w_last_block_txns; ++txn_id) {
        noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
            dram_noc_addr, w_dram_read_offset, slot_addr[slot_to_issue][txn_id], trid_to_issue);
        w_dram_read_offset += w_bytes_per_txn;
    }

    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w, w_tiles_per_block);

    ADVANCE_TRID(trid_to_wait);

    // Drain the pipeline - the last txn in flight
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w, w_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
