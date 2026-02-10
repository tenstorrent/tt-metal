// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "matmul_wo_ring_common.h"

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
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

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

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out = tt::CBIndex::c_2;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_out);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    const uint32_t num_tiles_h = matmul_wo_ring::K_TILES_PER_CORE_A[dram_bank_id];

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = matmul_wo_ring::W_TXNS_PER_BLOCK;
    constexpr uint32_t w_tiles_per_txn = matmul_wo_ring::W_TILES_PER_TXN;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;
    const uint32_t num_blocks_per_iter =
        (num_tiles_h * num_n_tiles_per_iter + w_tiles_per_block - 1) / w_tiles_per_block;
    const uint32_t w_total_blocks = num_blocks_per_iter * num_iters;
    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_bytes_per_block = w_tiles_per_block * w_tile_size;
    constexpr uint32_t w_bytes_per_txn = w_tiles_per_txn * w_tile_size;

    constexpr uint32_t w_max_blocks_per_iter = (max_num_tiles_h * num_n_tiles_per_iter) / w_tiles_per_block;
    constexpr uint32_t w_max_total_blocks = w_max_blocks_per_iter * num_iters;

    // Offsets for layer_id
    constexpr uint32_t w_size_per_layer = w_max_total_blocks * w_bytes_per_block;
    constexpr uint32_t w_layer_offset = layer_id * w_size_per_layer;

    // Offset for layer_id
    uint32_t w_dram_read_offset = w_layer_offset + w_addr;

    // DRAM bank's base NOC address
    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, /*bank_address_offset=*/0);

    //-------------------------------------------------------------------------
    // CB addresses
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = get_write_ptr(cb_r2c_w);

    // Precompute slot addresses (avoid multiply in hot loop)
    // Each slot holds 2 transactions (28 tiles)
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
    // Set w state once before loop (will be reused for all txns)
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
    for (uint32_t block_id = 0; block_id < w_total_blocks; ++block_id) {
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

    // Drain the pipeline - the last txn in flight
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_r2c_w, w_tiles_per_block);

    // We have one extra slot reserved, which we won't use.
    // For CB hygiene, we can push it back.
    cb_push_back(cb_r2c_w, w_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
