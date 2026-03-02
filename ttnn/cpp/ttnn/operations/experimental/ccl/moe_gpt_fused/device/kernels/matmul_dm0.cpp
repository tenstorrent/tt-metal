// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM0 (RISCV_1 / NOC_0) for moe_gpt_fused
//
// Phase 1: Exchange c_1 address with tilize drain core, wait for input.
// Phase 2: Pipelined weight reads from DRAM.

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
    const auto input_dram_addr = get_arg_val<uint32_t>(argidx++);       // unused
    const auto combine_semaphore_id = get_arg_val<uint32_t>(argidx++);  // unused by dm0
    const auto k_start_tile = get_arg_val<uint32_t>(argidx++);
    const auto output_base_l1_addr = get_arg_val<uint32_t>(argidx++);  // unused by dm0
    const auto tilize_ready_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto drain_noc_x = get_arg_val<uint32_t>(argidx++);
    const auto drain_noc_y = get_arg_val<uint32_t>(argidx++);
    const auto placeholder = get_arg_val<uint32_t>(argidx++);  // unused placeholder
    const auto addr_exchange_semaphore_id = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);

    // Constants
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_fused_ring::NUM_W0_W1_TILES_H;  // 90
    constexpr uint32_t num_w2_tiles_h = moe_gpt_fused_ring::NUM_W2_TILES_H;        // 90

    const uint32_t num_w0_w1_tiles_w = moe_gpt_fused_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_gpt_fused_ring::W2_TILES_PER_CORE_A[ring_core_id];

    //-------------------------------------------------------------------------
    // W0/W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_gpt_fused_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_gpt_fused_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile =
        4 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;
    constexpr uint32_t w0_w1_blocks_per_expert =
        w0_w1_blocks_per_two_elt_tile * moe_gpt_fused_ring::IN2_TILES_PER_STEP_A / 2;

    constexpr uint32_t w2_txns_per_block = moe_gpt_fused_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_gpt_fused_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;
    constexpr uint32_t w2_blocks_per_expert = moe_gpt_fused_ring::W2_BLOCKS_PER_EXPERT;

    constexpr uint32_t w0_w1_bytes_per_block = w0_w1_tiles_per_block * w0_w1_tile_size;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_w1_tile_size;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w0_w1_tile_size;

    constexpr uint32_t w0_size_per_expert = num_w0_w1_tiles_h * 8 * w0_w1_tile_size;
    constexpr uint32_t w0_w1_total_size_per_expert = 2 * w0_size_per_expert;
    constexpr uint32_t w0_w1_total_size_per_layer = num_experts * w0_w1_total_size_per_expert;
    constexpr uint32_t w0_w1_layer_offset = layer_id * w0_w1_total_size_per_layer;

    constexpr uint32_t w2_total_size_per_expert = num_w2_tiles_h * 8 * w0_w1_tile_size;
    constexpr uint32_t w2_total_size_per_layer = num_experts * w2_total_size_per_expert;
    constexpr uint32_t w2_layer_offset = layer_id * w2_total_size_per_layer;

    uint32_t w0_w1_expert_offset = w0_w1_layer_offset + w0_w1_addr;
    uint32_t w2_expert_offset = w2_layer_offset + w2_addr;

    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, 0);

    //-------------------------------------------------------------------------
    // Phase 1: Address exchange + wait for tilize cores to push input to c_1
    //
    // 1. Ring core 0 sends c_1 base address to drain core's addr_exchange semaphore
    // 2. All cores wait for tilize_ready signal (drain writes tiles to our c_1)
    //-------------------------------------------------------------------------
    {
        const uint32_t c1_base_addr = get_write_ptr(cb_s2c_in);

        // Only ring_core_id 0 sends the c_1 address to drain core
        // (all matmul cores have same c_1 address due to identical CB layout)
        if (ring_core_id == 0) {
            uint32_t addr_exchange_sem_local = get_semaphore(addr_exchange_semaphore_id);
            uint64_t drain_addr_exchange_noc = get_noc_addr(drain_noc_x, drain_noc_y, addr_exchange_sem_local);
            // Write the c_1 base address as the semaphore value
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr_exchange_sem_local), c1_base_addr);
            noc_semaphore_set_remote(addr_exchange_sem_local, drain_addr_exchange_noc);
        }

        // Wait for tilize drain core to signal that input has been written to our c_1
        uint32_t tilize_ready_sem_addr = get_semaphore(tilize_ready_semaphore_id);
        volatile tt_l1_ptr uint32_t* tilize_ready_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_ready_sem_addr);
        *tilize_ready_sem_ptr = 0;
        noc_semaphore_wait(tilize_ready_sem_ptr, 1);
    }

    //-------------------------------------------------------------------------
    // Phase 2: Read weights (same as moe_gpt dm0)
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = get_write_ptr(cb_r2c_w0_w1);
    const uint32_t slot_addr[NUM_SLOTS] = {
        w_cb_base_addr, w_cb_base_addr + w0_w1_bytes_per_block, w_cb_base_addr + 2 * w0_w1_bytes_per_block};

    noc_async_read_one_packet_set_state<true>(dram_noc_addr, w0_w1_bytes_per_txn, vchannel);

    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    bool txns_in_flight = false;

    cb_reserve_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
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
            cb_push_back(cb_r2c_w0_w1, w0_w1_tiles_per_block);
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
