// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/api/tt-metalium/constants.hpp"

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
    constexpr uint32_t collector_semaphore_id = get_named_compile_time_arg_val("collector_semaphore_id");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t first_physical_x = get_named_compile_time_arg_val("first_physical_x");
    constexpr uint32_t first_physical_y = get_named_compile_time_arg_val("first_physical_y");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_a_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto wq_b_args = TensorAccessorArgs<w_a_args.next_compile_time_args_offset()>();
    constexpr auto rope_args = TensorAccessorArgs<wq_b_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<rope_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto is_collector = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_a_addr = get_arg_val<uint32_t>(argidx++);
    const auto wq_b_addr = get_arg_val<uint32_t>(argidx++);
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
    constexpr uint32_t w_a_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t rope_tile_size = get_tile_size(cb_r2c_rope);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);

    // Constants for MLA WqkvAb
    constexpr uint32_t w_a_k_tiles = 7168 / tt::constants::TILE_WIDTH;
    constexpr uint32_t n_tiles_this_core = 6;
    constexpr uint32_t num_w_a_tiles = w_a_k_tiles * n_tiles_this_core;

    //-------------------------------------------------------------------------
    // W_a reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_a_txns_per_block = 6;
    constexpr uint32_t w_a_tiles_per_txn = 7;
    constexpr uint32_t w_a_tiles_per_block = w_a_tiles_per_txn * w_a_txns_per_block;
    constexpr uint32_t w_a_num_blocks = num_w_a_tiles / w_a_tiles_per_block;

    //-------------------------------------------------------------------------
    // Constants for MLA Wq_b
    constexpr uint32_t wq_b_k_tiles = 49;
    constexpr uint32_t wq_b_n_tiles_this_core = 8;
    constexpr uint32_t num_wq_b_tiles = wq_b_k_tiles * wq_b_n_tiles_this_core;
    //-------------------------------------------------------------------------
    // Wq-b reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t wq_b_txns_per_block = 4;
    constexpr uint32_t wq_b_tiles_per_txn = 7;
    constexpr uint32_t wq_b_tiles_per_block = wq_b_tiles_per_txn * wq_b_txns_per_block;
    constexpr uint32_t wq_b_num_blocks = num_wq_b_tiles / wq_b_tiles_per_block;

    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_a_bytes_per_block = w_a_tiles_per_block * w_a_tile_size;
    constexpr uint32_t w_a_bytes_per_txn = w_a_tiles_per_txn * w_a_tile_size;

    constexpr uint32_t wq_b_bytes_per_block = wq_b_tiles_per_block * w_a_tile_size;
    constexpr uint32_t wq_b_bytes_per_txn = wq_b_tiles_per_txn * w_a_tile_size;

    // DRAM bank's base NOC address
    const uint64_t w_a_noc_addr =
        get_noc_addr_from_bank_id</*DRAM=*/true>(dram_bank_id, /*bank_address_offset=*/w_a_addr);
    const uint64_t rope_noc_addr =
        get_noc_addr_from_bank_id</*DRAM=*/true>(dram_bank_id, /*bank_address_offset=*/rope_addr);
    const uint64_t wq_b_noc_addr =
        get_noc_addr_from_bank_id</*DRAM=*/true>(dram_bank_id, /*bank_address_offset=*/wq_b_addr);

    //-------------------------------------------------------------------------
    // Rope reading constants
    //-------------------------------------------------------------------------
    // We read just 6 rows of this tensor for a given position, which is 1/16 of the tile's rows
    constexpr uint32_t rope_elements_per_row = tt::constants::FACE_WIDTH;
    constexpr uint32_t rope_bytes_per_row = rope_elements_per_row * sizeof(uint16_t);  // bf16
    constexpr uint32_t rope_rows_per_pos = 2 * 2;                                      // 2 rows each for cos and sin
    constexpr uint32_t rope_rows_per_pos_pair = rope_rows_per_pos * 2;
    constexpr uint32_t rope_bytes_per_pos_pair = rope_rows_per_pos_pair * rope_bytes_per_row;
    constexpr uint32_t rope_bytes_per_txn = 6 * rope_bytes_per_row;  // Need 2 rows + 2 rows (dummy) + 2 rows

    const uint32_t rope_pos_pair_idx = pos / 2;
    const uint32_t rope_dram_pair_offset = rope_pos_pair_idx * rope_bytes_per_pos_pair;
    const uint32_t rope_dram_pos_offset = (pos & 1) * 2 * rope_bytes_per_row;
    const uint32_t rope_dram_read_offset = rope_dram_pair_offset + rope_dram_pos_offset;

    //-------------------------------------------------------------------------
    // CB addresses
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = get_write_ptr(cb_r2c_w);

    // Precompute slot addresses (avoid multiply in hot loop)
    // Each slot holds txns_per_block tiles
    uint32_t slot_addr[NUM_SLOTS][w_a_txns_per_block];

    uint32_t slot_addr_offset = 0;
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        for (uint32_t j = 0; j < w_a_txns_per_block; ++j) {
            slot_addr[i][j] = w_cb_base_addr + slot_addr_offset;
            slot_addr_offset += w_a_bytes_per_txn;
        }
    }

    //-------------------------------------------------------------------------
    // W reading loop
    //-------------------------------------------------------------------------
    // Set w_a state once before loop
    noc_async_read_one_packet_set_state<true>(w_a_noc_addr, w_a_bytes_per_txn, vchannel);

    //-------------------------------------------------------------------------
    // Variables to track pipeline state
    //-------------------------------------------------------------------------
    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    bool txns_in_flight = false;

    // We reserve one to kick start the pipeline, and then it is steady state
    cb_reserve_back(cb_r2c_w, w_a_tiles_per_block);

    //-------------------------------------------------------------------------
    // Pipelined reading of W
    //-------------------------------------------------------------------------
    uint32_t w_a_dram_read_offset = 0;

    for (uint32_t block_id = 0; block_id < w_a_num_blocks; ++block_id) {
        // Issue reads with current trid
        noc_async_read_set_trid(trid_to_issue);
        for (uint32_t txn_id = 0; txn_id < w_a_txns_per_block; ++txn_id) {
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                w_a_noc_addr, w_a_dram_read_offset, slot_addr[slot_to_issue][txn_id], trid_to_issue);
            w_a_dram_read_offset += w_a_bytes_per_txn;
        }

        ADVANCE_SLOT(slot_to_issue);
        ADVANCE_TRID(trid_to_issue);

        // Only when we first start the pipeline, we don't have any txns in flight
        if (txns_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_r2c_w, w_a_tiles_per_block);

            ADVANCE_TRID(trid_to_wait);

            // Reserve for next block
            // Reserve back is not incremental, so to reserve one more, we need to reserve 2
            // This accounts for the one we already have reserved (for in-flight read)
            cb_reserve_back(cb_r2c_w, w_a_tiles_per_block * 2);
        }
        txns_in_flight = true;
    }

    //-------------------------------------------------------------------------
    // Rope reading loop
    //-------------------------------------------------------------------------
    // Set rope state once before loop
    noc_async_read_one_packet_set_state<true>(rope_noc_addr, rope_bytes_per_txn, vchannel);
    cb_reserve_back(cb_r2c_rope, 1);

    noc_async_read_set_trid(trid_to_issue);
    noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
        rope_noc_addr, rope_dram_read_offset, get_write_ptr(cb_r2c_rope), trid_to_issue);

    ADVANCE_TRID(trid_to_issue);

    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w, w_a_tiles_per_block);

    ADVANCE_TRID(trid_to_wait);

    cb_reserve_back(cb_r2c_w, w_a_tiles_per_block * 2);

    //-------------------------------------------------------------------------
    // Pipelined reading of Wq_b
    //-------------------------------------------------------------------------
    // Use this as a flag to figure out which CB to push the data into
    // The first push goes to the RoPE CB, all others go to the cb_r2c_w CB
    bool wq_b_push_to_rope = true;
    uint32_t wq_b_dram_read_offset = 0;

    noc_async_read_one_packet_set_state<true>(wq_b_noc_addr, wq_b_bytes_per_txn, vchannel);

    for (uint32_t block_id = 0; block_id < wq_b_num_blocks; ++block_id) {
        // Issue reads with current trid
        noc_async_read_set_trid(trid_to_issue);
        for (uint32_t txn_id = 0; txn_id < wq_b_txns_per_block; ++txn_id) {
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                wq_b_noc_addr, wq_b_dram_read_offset, slot_addr[slot_to_issue][txn_id], trid_to_issue);
            wq_b_dram_read_offset += wq_b_bytes_per_txn;
        }

        ADVANCE_SLOT(slot_to_issue);
        ADVANCE_TRID(trid_to_issue);

        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(wq_b_push_to_rope ? cb_r2c_rope : cb_r2c_w, wq_b_push_to_rope ? 1 : w_a_tiles_per_block);

        ADVANCE_TRID(trid_to_wait);

        // Reserve for next block
        // Reserve back is not incremental, so to reserve one more, we need to reserve 2
        // This accounts for the one we already have reserved (for in-flight read)
        if (!wq_b_push_to_rope) {
            cb_reserve_back(cb_r2c_w, w_a_tiles_per_block * 2);
        }
        wq_b_push_to_rope = false;
    }

    // Drain the pipeline - the last txn in flight
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w, w_a_tiles_per_block);

    // We have one extra slot reserved, which we won't use.
    // For CB hygiene, we can push it back.
    cb_push_back(cb_r2c_w, w_a_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
