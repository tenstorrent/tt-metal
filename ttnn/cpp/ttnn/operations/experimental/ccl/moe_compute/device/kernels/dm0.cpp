// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    // Extract config type from compile-time argument
    constexpr uint32_t moe_config_type_value = get_named_compile_time_arg_val("moe_config_type");
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;

    constexpr auto config_type = static_cast<ttnn::experimental::prim::detail::MoEConfigType>(moe_config_type_value);

    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Ring is templatized on num_cores: 12 on Wormhole, 12 on Blackhole (BH pads from 8
    // DRAM-bank cores to 12 with INTERLEAVED weights). See moe_ring_common.h.
    using config_t = moe_ring::ConfigType_t<has_bias, config_type, num_cores>;

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
#if defined(ARCH_BLACKHOLE)
    // BH-only: starting global page_id of this ring core's slice in the INTERLEAVED weight
    // buffer (in tiles). On WH the weight buffers are HEIGHT_SHARDED so each core's slice
    // lives in its own dedicated DRAM bank; the start_page_id concept doesn't apply.
    const auto w0_w1_core_start_page_id = get_arg_val<uint32_t>(argidx++);
    const auto w2_core_start_page_id = get_arg_val<uint32_t>(argidx++);
#endif

    // CBs
    constexpr auto cb_s2c_in = tt::CBIndex::c_0;     // tilize_output_cb_id
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_3;  // cb_r2c_w0
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_4;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_5;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_md = tt::CBIndex::c_7;

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;  // matmul_writer_cb_id
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_3;   // reuse cb_r2c_w0_w1

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_ring::W0_W1_TILES_PER_TXN;
    // constexpr uint32_t w0_w1_tiles_w = moe_ring::W0_W1_BLOCK_TILES_W;
    constexpr uint32_t w0_w1_block_tiles_h = moe_ring::W0_W1_BLOCK_TILES_H;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28

    constexpr uint32_t w0_w1_dram_tiles_h = config_t::NUM_W0_W1_DRAM_TILES_H;
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile = detail::div_up<w0_w1_dram_tiles_h, w0_w1_block_tiles_h>();
    constexpr uint32_t w0_w1_blocks_per_expert = w0_w1_blocks_per_two_elt_tile * config_t::IN2_TILES_PER_STEP / 2;

    // W2 reading constants
    constexpr uint32_t w2_dram_tiles_h = config_t::NUM_W2_DRAM_TILES_H;
    constexpr uint32_t w2_txns_per_block = moe_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;               // 14 * 2 = 28
    constexpr uint32_t w2_txns_h = (w2_dram_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;
    constexpr uint32_t w2_blocks_per_expert = config_t::W2_BLOCKS_PER_EXPERT;

    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_bytes_per_block = w0_w1_tiles_per_block * w0_w1_tile_size;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_w1_tile_size;
    constexpr uint32_t w2_bytes_per_block = w2_tiles_per_block * w2_tile_size;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w2_tile_size;

    // Offsets for layer_id

    constexpr uint32_t w0_w1_total_size_per_expert = w0_w1_blocks_per_expert * 2 * w0_w1_bytes_per_txn;
    constexpr uint32_t w0_w1_total_size_per_layer = num_experts * w0_w1_total_size_per_expert;
    constexpr uint32_t w0_w1_layer_offset = layer_id * w0_w1_total_size_per_layer;

    // W2: same approach
    constexpr uint32_t w2_total_size_per_expert = w2_blocks_per_expert * 2 * w2_bytes_per_txn;
    constexpr uint32_t w2_total_size_per_layer = num_experts * w2_total_size_per_expert;
    constexpr uint32_t w2_layer_offset = layer_id * w2_total_size_per_layer;

#if defined(ARCH_BLACKHOLE)
    // BH: weights are INTERLEAVED across DRAM banks (BH has 8 banks but ring uses 12 cores).
    // TensorAccessor pages tiles by global page_id; bank selection is automatic.
    // Tile counts (per core) for layer/expert striding in tile units.
    constexpr uint32_t w0_w1_pages_per_expert = w0_w1_total_size_per_expert / w0_w1_tile_size;
    constexpr uint32_t w0_w1_pages_per_layer = w0_w1_total_size_per_layer / w0_w1_tile_size;
    constexpr uint32_t w0_w1_layer_page_offset = layer_id * w0_w1_pages_per_layer;
    constexpr uint32_t w0_w1_pages_per_txn = w0_w1_tiles_per_txn;

    constexpr uint32_t w2_pages_per_expert = w2_total_size_per_expert / w2_tile_size;
    constexpr uint32_t w2_pages_per_layer = w2_total_size_per_layer / w2_tile_size;
    constexpr uint32_t w2_layer_page_offset = layer_id * w2_pages_per_layer;
    constexpr uint32_t w2_pages_per_txn = w2_tiles_per_txn;

    // TensorAccessors for INTERLEAVED weight buffers. Page size = tile size.
    const auto w0_w1_acc = TensorAccessor(w0_w1_args, w0_w1_addr, w0_w1_tile_size);
    const auto w2_acc = TensorAccessor(w2_args, w2_addr, w2_tile_size);

    // Starting global page_id for this ring core's first expert in the current layer
    uint32_t w0_w1_expert_page_offset = w0_w1_core_start_page_id + w0_w1_layer_page_offset;
    uint32_t w2_expert_page_offset = w2_core_start_page_id + w2_layer_page_offset;
    // Mark unused for BH
    (void)dram_bank_id;
    (void)vchannel;
#else
    // Offsets for expert_id (WH HEIGHT_SHARDED — byte offsets within this core's bank slice)
    uint32_t w0_w1_expert_offset = w0_w1_layer_offset + w0_w1_addr;
    uint32_t w2_expert_offset = w2_layer_offset + w2_addr;

    // DRAM bank's base NOC address
    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, /*bank_address_offset=*/0);
#endif

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
#if !defined(ARCH_BLACKHOLE)
    // WH only: set w0_w1 state once before loop (will be reused for all experts).
    // BH path issues per-tile reads and cannot reuse a single base address.
    noc_async_read_one_packet_set_state<true>(dram_noc_addr, w0_w1_bytes_per_txn, vchannel);
#endif

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

    // Read per-expert token counts from CB
    volatile tt_l1_ptr uint32_t* num_tokens_per_expert_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(per_expert_total_tokens_cb_id));

    // Precompute NUM_CHUNKS_PER_EXPERT
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = num_tokens_per_expert_ptr[expert_id];
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
#if defined(ARCH_BLACKHOLE)
            // BH: per-tile reads via TensorAccessor (INTERLEAVED layout).
            uint32_t w0_w1_page_id = w0_w1_expert_page_offset;
#else
            uint32_t w0_w1_dram_read_offset = w0_w1_expert_offset;
#endif

            for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_expert; ++block_id) {
                // Issue reads with current trid (set_trid persists in NOC_PACKET_TAG cmd_buf
                // register; subsequent fast_reads inherit the trid until next set_trid).
                noc_async_read_set_trid(trid_to_issue);
#if defined(ARCH_BLACKHOLE)
                // BH: 2 transactions per block, each = w0_w1_pages_per_txn (=14) per-tile reads.
                // Each tile is on a (potentially) different bank — TensorAccessor::get_noc_addr
                // resolves bank automatically. trid is reused across all per-tile issues until
                // the next set_trid call.
                {
                    uint32_t l1_dst_addr = slot_addr[slot_to_issue];
                    for (uint32_t t = 0; t < w0_w1_pages_per_txn; ++t) {
                        noc_async_read_one_packet(w0_w1_acc.get_noc_addr(w0_w1_page_id), l1_dst_addr, w0_w1_tile_size);
                        l1_dst_addr += w0_w1_tile_size;
                        ++w0_w1_page_id;
                    }
                }
                {
                    uint32_t l1_dst_addr = slot_addr[slot_to_issue] + w0_w1_bytes_per_txn;
                    for (uint32_t t = 0; t < w0_w1_pages_per_txn; ++t) {
                        noc_async_read_one_packet(w0_w1_acc.get_noc_addr(w0_w1_page_id), l1_dst_addr, w0_w1_tile_size);
                        l1_dst_addr += w0_w1_tile_size;
                        ++w0_w1_page_id;
                    }
                }
#else
                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr, w0_w1_dram_read_offset, slot_addr[slot_to_issue], trid_to_issue);
                w0_w1_dram_read_offset += w0_w1_bytes_per_txn;

                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr,
                    w0_w1_dram_read_offset,
                    slot_addr[slot_to_issue] + w0_w1_bytes_per_txn,
                    trid_to_issue);
                w0_w1_dram_read_offset += w0_w1_bytes_per_txn;
#endif

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
#if defined(ARCH_BLACKHOLE)
            uint32_t w2_page_id = w2_expert_page_offset;
#else
            uint32_t w2_dram_read_offset = w2_expert_offset;
#endif

            for (uint32_t block_id = 0; block_id < w2_blocks_per_expert; ++block_id) {
                // Issue reads with current trid
                noc_async_read_set_trid(trid_to_issue);
#if defined(ARCH_BLACKHOLE)
                {
                    uint32_t l1_dst_addr = slot_addr[slot_to_issue];
                    for (uint32_t t = 0; t < w2_pages_per_txn; ++t) {
                        noc_async_read_one_packet(w2_acc.get_noc_addr(w2_page_id), l1_dst_addr, w2_tile_size);
                        l1_dst_addr += w2_tile_size;
                        ++w2_page_id;
                    }
                }
                {
                    uint32_t l1_dst_addr = slot_addr[slot_to_issue] + w2_bytes_per_txn;
                    for (uint32_t t = 0; t < w2_pages_per_txn; ++t) {
                        noc_async_read_one_packet(w2_acc.get_noc_addr(w2_page_id), l1_dst_addr, w2_tile_size);
                        l1_dst_addr += w2_tile_size;
                        ++w2_page_id;
                    }
                }
#else
                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr, w2_dram_read_offset, slot_addr[slot_to_issue], trid_to_issue);
                w2_dram_read_offset += w2_bytes_per_txn;

                noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                    dram_noc_addr, w2_dram_read_offset, slot_addr[slot_to_issue] + w2_bytes_per_txn, trid_to_issue);
                w2_dram_read_offset += w2_bytes_per_txn;
#endif

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
#if defined(ARCH_BLACKHOLE)
        w0_w1_expert_page_offset += w0_w1_pages_per_expert;
        w2_expert_page_offset += w2_pages_per_expert;
#else
        w0_w1_expert_offset += w0_w1_total_size_per_expert;
        w2_expert_offset += w2_total_size_per_expert;
#endif
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
