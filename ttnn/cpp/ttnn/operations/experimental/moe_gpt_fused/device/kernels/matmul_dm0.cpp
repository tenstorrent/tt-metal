// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM0 (RISCV_1 / NOC_0) for moe_gpt_fused
//
// Same as moe_gpt dm0 but also reads input from DRAM into c_1 before
// starting weight reads. Uses named compile args instead of TensorAccessorArgs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_fused_ring_common.h"

// Triple buffering for weight reads
#define NUM_SLOTS 3
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
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Runtime args
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto input_dram_addr = get_arg_val<uint32_t>(argidx++);
    const auto output_dram_addr = get_arg_val<uint32_t>(argidx++);
    const auto k_start_tile = get_arg_val<uint32_t>(argidx++);

    // CBs (same as moe_gpt)
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);

    // Constants
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_fused_ring::NUM_W0_W1_TILES_H;  // 90
    constexpr uint32_t num_w2_tiles_h = moe_gpt_fused_ring::NUM_W2_TILES_H;        // 90

    const uint32_t num_w0_w1_tiles_w = moe_gpt_fused_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];  // 7 or 8
    const uint32_t num_w2_tiles_w = moe_gpt_fused_ring::W2_TILES_PER_CORE_A[ring_core_id];                    // 7 or 8

    //-------------------------------------------------------------------------
    // W0/W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_gpt_fused_ring::W0_W1_TXNS_PER_BLOCK;     // 2
    constexpr uint32_t w0_w1_tiles_per_txn = moe_gpt_fused_ring::W0_W1_TILES_PER_TXN;       // 10
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 20
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile =
        4 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;  // 18
    constexpr uint32_t w0_w1_blocks_per_expert =
        w0_w1_blocks_per_two_elt_tile * moe_gpt_fused_ring::IN2_TILES_PER_STEP_A / 2;  // 72

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = moe_gpt_fused_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_gpt_fused_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;        // 20
    constexpr uint32_t w2_blocks_per_expert = moe_gpt_fused_ring::W2_BLOCKS_PER_EXPERT;  // 36

    // DRAM byte sizes
    constexpr uint32_t w0_w1_bytes_per_block = w0_w1_tiles_per_block * w0_w1_tile_size;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_w1_tile_size;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w0_w1_tile_size;  // w2 has same tile size as w0/w1

    // Layer offsets (same as moe_gpt)
    constexpr uint32_t w0_size_per_expert = num_w0_w1_tiles_h * 8 * w0_w1_tile_size;
    constexpr uint32_t w0_w1_total_size_per_expert = 2 * w0_size_per_expert;
    constexpr uint32_t w0_w1_total_size_per_layer = num_experts * w0_w1_total_size_per_expert;
    constexpr uint32_t w0_w1_layer_offset = layer_id * w0_w1_total_size_per_layer;

    constexpr uint32_t w2_total_size_per_expert = num_w2_tiles_h * 8 * w0_w1_tile_size;
    constexpr uint32_t w2_total_size_per_layer = num_experts * w2_total_size_per_expert;
    constexpr uint32_t w2_layer_offset = layer_id * w2_total_size_per_layer;

    uint32_t w0_w1_expert_offset = w0_w1_layer_offset + w0_w1_addr;
    uint32_t w2_expert_offset = w2_layer_offset + w2_addr;

    // DRAM bank NOC address
    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, 0);

    //-------------------------------------------------------------------------
    // Phase 1: Read input tiles from DRAM into c_1[0..89]
    // Single copy shared by all experts (no per-expert duplication)
    //-------------------------------------------------------------------------
    {
        const uint32_t cb_base = get_write_ptr(cb_s2c_in);

        const InterleavedAddrGen<true> input_addrgen = {
            .bank_base_address = input_dram_addr,
            .page_size = in_tile_size,
        };

        for (uint32_t t = 0; t < num_w0_w1_tiles_h; ++t) {
            uint32_t l1_addr = cb_base + t * in_tile_size;
            noc_async_read_tile(t, input_addrgen, l1_addr);
        }
        noc_async_read_barrier();
    }

    //-------------------------------------------------------------------------
    // Phase 2: Read weights (same as moe_gpt dm0)
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = get_write_ptr(cb_r2c_w0_w1);
    const uint32_t slot_addr[NUM_SLOTS] = {
        w_cb_base_addr, w_cb_base_addr + w0_w1_bytes_per_block, w_cb_base_addr + 2 * w0_w1_bytes_per_block};

    // Set NOC state for weight reads
    noc_async_read_one_packet_set_state<true>(dram_noc_addr, w0_w1_bytes_per_txn, vchannel);

    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    bool txns_in_flight = false;

    cb_reserve_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Pipelined W0/W1 reads
        uint32_t w0_w1_dram_read_offset = w0_w1_expert_offset;

        for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_expert; ++block_id) {
            noc_async_read_set_trid(trid_to_issue);
            noc_async_read_one_packet_with_state_with_trid<false, true>(
                dram_noc_addr, w0_w1_dram_read_offset, slot_addr[slot_to_issue], trid_to_issue);
            w0_w1_dram_read_offset += w0_w1_bytes_per_txn;

            noc_async_read_one_packet_with_state_with_trid<false, true>(
                dram_noc_addr, w0_w1_dram_read_offset, slot_addr[slot_to_issue] + w0_w1_bytes_per_txn, trid_to_issue);
            w0_w1_dram_read_offset += w0_w1_bytes_per_txn;

            ADVANCE_SLOT(slot_to_issue);
            ADVANCE_TRID(trid_to_issue);

            if (txns_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);
                ADVANCE_TRID(trid_to_wait);
                cb_reserve_back(cb_r2c_w0_w1, w0_w1_tiles_per_block * 2);
            }
            txns_in_flight = true;
        }

        // Pipelined W2 reads
        uint32_t w2_dram_read_offset = w2_expert_offset;

        for (uint32_t block_id = 0; block_id < w2_blocks_per_expert; ++block_id) {
            noc_async_read_set_trid(trid_to_issue);
            noc_async_read_one_packet_with_state_with_trid<false, true>(
                dram_noc_addr, w2_dram_read_offset, slot_addr[slot_to_issue], trid_to_issue);
            w2_dram_read_offset += w2_bytes_per_txn;

            noc_async_read_one_packet_with_state_with_trid<false, true>(
                dram_noc_addr, w2_dram_read_offset, slot_addr[slot_to_issue] + w2_bytes_per_txn, trid_to_issue);
            w2_dram_read_offset += w2_bytes_per_txn;

            ADVANCE_SLOT(slot_to_issue);
            ADVANCE_TRID(trid_to_issue);

            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);  // cb_r2c_w2 aliases cb_r2c_w0_w1
            ADVANCE_TRID(trid_to_wait);
            cb_reserve_back(cb_r2c_w0_w1, w0_w1_tiles_per_block * 2);
        }

        w0_w1_expert_offset += w0_w1_total_size_per_expert;
        w2_expert_offset += w2_total_size_per_expert;
    }

    // Drain pipeline
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);
    cb_push_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
