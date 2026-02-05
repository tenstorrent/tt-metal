// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_ring_common.h"

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
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // For synchronization with tilize cores
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    constexpr auto w0_w1_args = TensorAccessorArgs<0>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_s2c_in = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;
    constexpr auto cb_w2c_md = tt::CBIndex::c_5;

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_1;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = moe_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_ring::W2_TILES_PER_CORE_A[ring_core_id];

    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile =
        4 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;  // 32
    constexpr uint32_t w0_w1_blocks_per_expert =
        w0_w1_blocks_per_two_elt_tile * moe_ring::IN2_TILES_PER_STEP_A / 2;  // 32 * 3 = 96
    // 2 * num_w0_w1_tiles_w * num_w0_w1_tiles_h / w0_w1_tiles_per_block;  // (5|6 * 224) / 28 = 80|96

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = moe_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;               // 14 * 2 = 28
    constexpr uint32_t w2_txns_h = (num_w2_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;  // 5 (round up)
    constexpr uint32_t w2_blocks_per_four_mm2_tile = 4 * w2_txns_h / w2_txns_per_block;         // 4 * 5 / 2 = 10
    constexpr uint32_t w2_blocks_per_expert = moe_ring::W2_BLOCKS_PER_EXPERT;

    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_bytes_per_block = w0_w1_tiles_per_block * w0_w1_tile_size;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_w1_tile_size;
    constexpr uint32_t w2_bytes_per_block = w2_tiles_per_block * w2_tile_size;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w2_tile_size;

    // Offsets for layer_id
    constexpr uint32_t w0_size_per_expert = num_w0_w1_tiles_h * 6 * w0_w1_tile_size;
    constexpr uint32_t w0_w1_total_size_per_expert = 2 * w0_size_per_expert;
    constexpr uint32_t w0_w1_total_size_per_layer = num_experts * w0_w1_total_size_per_expert;
    constexpr uint32_t w0_w1_layer_offset = layer_id * w0_w1_total_size_per_layer;

    constexpr uint32_t w2_total_size_per_expert = 70 * 20 * w2_tile_size;  // We pad 64 to 70 tiles
    constexpr uint32_t w2_total_size_per_layer = num_experts * w2_total_size_per_expert;
    constexpr uint32_t w2_layer_offset = layer_id * w2_total_size_per_layer;

    // Offsets for expert_id
    uint32_t w0_w1_expert_offset = w0_w1_layer_offset + w0_w1_addr;
    uint32_t w2_expert_offset = w2_layer_offset + w2_addr;

    // DRAM bank's base NOC address
    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, /*bank_address_offset=*/0);

    //-------------------------------------------------------------------------
    // CB addresses
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = get_write_ptr(cb_r2c_w0_w1);

    // Precompute slot addresses (avoid multiply in hot loop)
    // Each slot holds 2 transactions (28 tiles)
    const uint32_t slot_addr[NUM_SLOTS] = {
        w_cb_base_addr, w_cb_base_addr + w0_w1_bytes_per_block, w_cb_base_addr + 2 * w0_w1_bytes_per_block};

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    // Set w0_w1 state once before loop (will be reused for all experts)
    noc_async_read_one_packet_set_state<true>(dram_noc_addr, w0_w1_bytes_per_txn, vchannel);

    //-------------------------------------------------------------------------
    // Variables to track pipeline state
    //-------------------------------------------------------------------------
    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    bool txns_in_flight = false;

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Receive number of tokens per expert from the tilize cores
    uint32_t metadata_ready_semaphore_addr = get_semaphore(metadata_ready_semaphore_id);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_ready_semaphore_addr), 1);

    // Precompute NUM_CHUNKS_PER_EXPERT
    // NOTE: hardcoded to 2 experts
    volatile tt_l1_ptr uint32_t* metadata_ready_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(metadata_ready_semaphore_id));
    uint32_t encoded_metadata_value = *metadata_ready_semaphore_ptr;

    constexpr uint32_t BITS_PER_EXPERT = 10;
    constexpr uint32_t EXPERT_MASK = 0x3FFu;
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = (encoded_metadata_value >> (1 + BITS_PER_EXPERT * expert_id)) & EXPERT_MASK;
        NUM_CHUNKS_PER_EXPERT[expert_id] = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    }

    //-------------------------------------------------------------------------
    // Start pipeline
    //-------------------------------------------------------------------------

    // We reserve one to kick start the pipeline, and then it is steady state
    cb_reserve_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];
        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            //-------------------------------------------------------------------------
            // Pipelined reading of W0/W1
            //-------------------------------------------------------------------------
            uint32_t w0_w1_dram_read_offset = w0_w1_expert_offset;

            for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_expert; ++block_id) {
                // Issue reads with current trid
                noc_async_read_set_trid(trid_to_issue);
                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr, w0_w1_dram_read_offset, slot_addr[slot_to_issue], trid_to_issue);
                w0_w1_dram_read_offset += w0_w1_bytes_per_txn;

                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr,
                    w0_w1_dram_read_offset,
                    slot_addr[slot_to_issue] + w0_w1_bytes_per_txn,
                    trid_to_issue);
                w0_w1_dram_read_offset += w0_w1_bytes_per_txn;

                ADVANCE_SLOT(slot_to_issue);
                ADVANCE_TRID(trid_to_issue);

                // Only when we first start the pipeline, we don't have any txns in flight
                if (txns_in_flight) {
                    noc_async_read_barrier_with_trid(trid_to_wait);
                    cb_push_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);

                    ADVANCE_TRID(trid_to_wait);

                    // Reserve for next block
                    // Reserve back is not incremental, so to reserve one more, we need to reserve 2
                    // This accounts for the one we already have reserved (for in-flight read)
                    cb_reserve_back(cb_r2c_w0_w1, w0_w1_tiles_per_block * 2);
                }
                txns_in_flight = true;
            }

            //-------------------------------------------------------------------------
            // Pipelined reading of W2
            //-------------------------------------------------------------------------
            uint32_t w2_dram_read_offset = w2_expert_offset;

            for (uint32_t block_id = 0; block_id < w2_blocks_per_expert; ++block_id) {
                // Issue reads with current trid
                noc_async_read_set_trid(trid_to_issue);
                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr, w2_dram_read_offset, slot_addr[slot_to_issue], trid_to_issue);
                w2_dram_read_offset += w2_bytes_per_txn;

                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr, w2_dram_read_offset, slot_addr[slot_to_issue] + w2_bytes_per_txn, trid_to_issue);
                w2_dram_read_offset += w2_bytes_per_txn;

                ADVANCE_SLOT(slot_to_issue);
                ADVANCE_TRID(trid_to_issue);

                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_r2c_w2, w2_tiles_per_block);

                ADVANCE_TRID(trid_to_wait);

                // Reserve for next block
                // Reserve back is not incremental, so to reserve one more, we need to reserve 2
                // This accounts for the one we already have reserved (for in-flight read)
                cb_reserve_back(cb_r2c_w2, w2_tiles_per_block * 2);
            }
        }

        // Update offsets for next expert
        w0_w1_expert_offset += w0_w1_total_size_per_expert;
        w2_expert_offset += w2_total_size_per_expert;
    }

    // Drain the pipeline - the last txn in flight
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w2, w2_tiles_per_block);

    // We have one extra slot reserved, which we won't use.
    // For CB hygiene, we can push it back.
    cb_push_back(cb_r2c_w2, w2_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
