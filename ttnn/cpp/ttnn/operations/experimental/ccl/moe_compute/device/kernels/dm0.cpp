// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
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
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t Ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_shared_experts = get_named_compile_time_arg_val("num_shared_experts");
    constexpr uint32_t shared_expert_tp_factor = get_named_compile_time_arg_val("shared_expert_tp_factor");

    using Cfg = moe_ring::MoeRingConfig<Ht, Nt, num_cores, has_bias, shared_expert_tp_factor>;

    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    // Number of physical DRAM banks the HEIGHT_SHARDED weight tensor lives on. WH=12 (1:1
    // with ring N=12). BH=8 always; ring N can be 8, 12, or 16. When N exceeds num_banks
    // each ring core's slice is smaller than a bank, but may straddle one bank boundary;
    // the bank-run loop below handles up to two banks per slice.
    constexpr uint32_t num_banks = get_named_compile_time_arg_val("num_banks");
    // Per-ring-core total tile-page count (across ALL layers and ALL experts). Derived from
    // the HEIGHT_SHARDED weight tensor's total page count divided by num_cores (the prepare
    // function emits a leading dim of num_cores; HEIGHT_SHARDED keeps the byte order so the
    // flat layout is core-major).
    constexpr uint32_t w0_w1_pages_per_ring_core_total =
        get_named_compile_time_arg_val("w0_w1_pages_per_ring_core_total");
    constexpr uint32_t w2_pages_per_ring_core_total = get_named_compile_time_arg_val("w2_pages_per_ring_core_total");

    // For synchronization with tilize cores
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    constexpr auto w0_w1_args = TensorAccessorArgs<0>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments. dm0 and dm1 share one rt-arg layout emitted by the host
    // (matmul_runtime_args in program_factory.cpp), so dm0 has to consume the layout
    // positions it doesn't use. Args used here: vchannel, w0_w1_addr, w2_addr, ring_core_id.
    // The rest are dm1-only (out_addr, ring_semaphore_id, ring_neighbor_physical_*) or
    // legacy placeholders (dram_bank_id).
    uint32_t argidx = 0;
    [[maybe_unused]] const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto out_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // shard_to_bank translation table: maps shard index -> physical chip DRAM bank id.
    // The host appends `num_banks` entries here. The bank-run loop below reads its
    // shard_idx from `gp / pages_per_bank_total`, then translates via this table to get
    // the actual chip bank to feed `get_noc_addr_from_bank_id`.
    uint32_t shard_to_bank[num_banks];
    for (uint32_t i = 0; i < num_banks; ++i) {
        shard_to_bank[i] = get_arg_val<uint32_t>(argidx++);
    }

    // CBs
    constexpr auto cb_s2c_in_id = tt::CBIndex::c_0;     // tilize_output_cb_id
    constexpr auto cb_r2c_w0_w1_id = tt::CBIndex::c_3;  // cb_r2c_w0
    constexpr auto cb_c2w_rdy_id = tt::CBIndex::c_4;
    constexpr auto cb_w2c_rdy_id = tt::CBIndex::c_5;
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_6;
    constexpr auto cb_w2c_md_id = tt::CBIndex::c_7;

    // CB Aliases
    constexpr auto cb_c2s_out_id = tt::CBIndex::c_1;  // matmul_writer_cb_id
    constexpr auto cb_r2c_w2_id = tt::CBIndex::c_3;   // reuse cb_r2c_w0_w1

    // CircularBuffer typed wrappers
    CircularBuffer cb_r2c_w0_w1(cb_r2c_w0_w1_id);
    CircularBuffer cb_r2c_w2(cb_r2c_w2_id);
    CircularBuffer cb_per_expert_total_tokens(per_expert_total_tokens_cb_id);

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in_id);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1_id);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2_id);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2_id);

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = moe_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;  // 14 * 2 = 28

    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_bytes_per_block = w0_w1_tiles_per_block * w0_w1_tile_size;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_w1_tile_size;
    [[maybe_unused]] constexpr uint32_t w2_bytes_per_block = w2_tiles_per_block * w2_tile_size;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w2_tile_size;

    // Bank-run loop invariant: w0_w1 and w2 each track their own cur_shard_idx_* but share
    // the same physical NoC cmd-buf size register via noc_async_read_one_packet_set_state.
    // The sentinel init of cur_shard_idx_w2 below forces a fresh set_state only on the FIRST
    // w2 read; subsequent stream transitions can land on a matching cur_shard_idx_w2 while
    // the cmd-buf still holds w0_w1's size. As long as both streams use the same bytes_per_txn
    // the size-state collision is benign. If a future config diverges these sizes, either
    // invalidate cur_shard_idx_* at every stream boundary or move to per-stream cmd-bufs.
    static_assert(
        w0_w1_bytes_per_txn == w2_bytes_per_txn,
        "Bank-run loop assumes w0_w1 and w2 share identical bytes_per_txn (NoC cmd-buf size).");

    // Tile-count math for the bank-run loop. The HEIGHT_SHARDED weight tensor stores the
    // FULL flat tile sequence of the prepare-output tensor, whose leading dim is `num_cores`.
    // The flat layout is therefore CORE-MAJOR: ring_core_0's tiles for all (layer, expert),
    // then ring_core_1's tiles, etc. HEIGHT_SHARDED splits this flat sequence into
    // `num_banks` equal chunks (one per physical DRAM bank).
    //
    //   pages_per_logical_shard       = blocks_per_expert * tiles_per_block
    //                                       (this ring core's tiles for ONE expert in ONE layer)
    //   pages_per_ring_core_total     = num_layers * num_experts * pages_per_logical_shard
    //                                       (computed host-side from buffer->num_pages() / num_cores)
    //   pages_per_bank_total          = num_cores * pages_per_ring_core_total / num_banks
    //
    // For ring core r, layer l, expert e, the slice's first global page is
    //     gp = r * pages_per_ring_core_total + l * num_experts * pages_per_logical_shard
    //          + e * pages_per_logical_shard
    // and it runs for `pages_per_logical_shard` consecutive pages.
    //
    // For a global page id `gp`:
    //     shard_idx    = gp / pages_per_bank_total
    //     in_bank_page = gp - shard_idx * pages_per_bank_total
    // `shard_idx` is the placement-order index in [0, num_banks); the chip bank id is
    // obtained via `shard_to_bank[shard_idx]` (host computes this from the actual
    // buffer placement returned by `buffer()->get_buffer_page_mapping()`).
    constexpr uint32_t w0_w1_pages_per_logical_shard = Cfg::w0_w1_blocks_per_expert * w0_w1_tiles_per_block;
    constexpr uint32_t w0_w1_pages_total = num_cores * w0_w1_pages_per_ring_core_total;
    static_assert(w0_w1_pages_total % num_banks == 0, "w0_w1 pages_total must be divisible by num_banks");
    constexpr uint32_t w0_w1_pages_per_bank_total = w0_w1_pages_total / num_banks;
    // Each transaction is `tiles_per_txn` (=14) contiguous tiles. For the bank-run to work
    // without splitting a single transaction across a bank boundary, both the slice size and
    // the bank size must be multiples of the transaction tile count.
    static_assert(
        w0_w1_pages_per_logical_shard % w0_w1_tiles_per_txn == 0,
        "w0_w1 pages_per_logical_shard must be a multiple of tiles_per_txn (no mid-txn bank split allowed)");
    static_assert(
        w0_w1_pages_per_bank_total % w0_w1_tiles_per_txn == 0,
        "w0_w1 pages_per_bank_total must be a multiple of tiles_per_txn");

    constexpr uint32_t w2_pages_per_logical_shard = Cfg::w2_blocks_per_expert * w2_tiles_per_block;
    constexpr uint32_t w2_pages_total = num_cores * w2_pages_per_ring_core_total;
    static_assert(w2_pages_total % num_banks == 0, "w2 pages_total must be divisible by num_banks");
    constexpr uint32_t w2_pages_per_bank_total = w2_pages_total / num_banks;
    static_assert(
        w2_pages_per_logical_shard % w2_tiles_per_txn == 0,
        "w2 pages_per_logical_shard must be a multiple of tiles_per_txn");
    static_assert(
        w2_pages_per_bank_total % w2_tiles_per_txn == 0, "w2 pages_per_bank_total must be a multiple of tiles_per_txn");

    // Layer's per-ring-core stride in pages.
    constexpr uint32_t w0_w1_layer_pages_per_ring_core = num_experts * w0_w1_pages_per_logical_shard;
    constexpr uint32_t w2_layer_pages_per_ring_core = num_experts * w2_pages_per_logical_shard;
    constexpr uint32_t w0_w1_layer_offset_in_ring_core = layer_id * w0_w1_layer_pages_per_ring_core;
    constexpr uint32_t w2_layer_offset_in_ring_core = layer_id * w2_layer_pages_per_ring_core;

    //-------------------------------------------------------------------------
    // CB addresses
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = cb_r2c_w0_w1.get_write_ptr();

    // Precompute slot addresses (avoid multiply in hot loop)
    // Each slot holds 2 transactions (28 tiles)
    const uint32_t slot_addr[NUM_SLOTS] = {
        w_cb_base_addr, w_cb_base_addr + w0_w1_bytes_per_block, w_cb_base_addr + 2 * w0_w1_bytes_per_block};

    //-------------------------------------------------------------------------
    // Variables to track pipeline state
    //-------------------------------------------------------------------------
    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    bool txns_in_flight = false;

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Receive number of tokens per expert from the tilize cores
    Semaphore<> metadata_ready_sem(metadata_ready_semaphore_id);
    metadata_ready_sem.wait_min(1);

    // Read per-expert token counts from CB
    volatile tt_l1_ptr uint32_t* num_tokens_per_expert_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_per_expert_total_tokens.get_read_ptr());

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
    cb_r2c_w0_w1.reserve_back(w0_w1_tiles_per_block);

    // Pre-set state for this ring core's first bank (WH fast path: when
    // pages_per_ring_core_total <= pages_per_bank_total). The bank-run loop below will
    // re-set_state only when shard_idx changes.
    // Device 2.0 migration: legacy primitives retained: noc_async_read_set_trid /
    // noc_async_read_one_packet_set_state / noc_async_read_one_packet_with_state_with_trid /
    // noc_async_read_barrier_with_trid are the trid-pipelined state-machine API used to
    // drive a triple-buffered DRAM read pipeline; Device 2.0 Noc wrapper does not yet expose
    // typed equivalents for the set_state / with_state / with_trid family
    const uint32_t initial_shard_idx_w0 =
        (ring_core_id * w0_w1_pages_per_ring_core_total + w0_w1_layer_offset_in_ring_core) / w0_w1_pages_per_bank_total;
    const uint32_t initial_bank_id_w0 = shard_to_bank[initial_shard_idx_w0];
    {
        const uint64_t initial_dram_noc_addr_w0 = get_noc_addr_from_bank_id<true>(initial_bank_id_w0, 0);
        noc_async_read_one_packet_set_state<true>(initial_dram_noc_addr_w0, w0_w1_bytes_per_txn, vchannel);
    }
    // Track the currently set_state'd bank for w0_w1 reads. Initially equal to the bank we
    // just set above. The bank-run loop only re-set_states if shard_idx changes.
    uint32_t cur_shard_idx_w0 = initial_shard_idx_w0;
    // Same for w2 reads. Per-stream cache backed by ONE physical cmd-buf -- see the
    // static_assert at the top of the kernel that locks the bytes_per_txn invariant.
    // Init to a sentinel so the first w2 transaction always re-sets the cmd-buf (cheap
    // insurance; the static_assert makes inheriting w0_w1's state correctness-safe too).
    uint32_t cur_shard_idx_w2 = num_banks;

    // This ring core's first global page id for the CURRENT layer.
    const uint32_t w0_w1_ring_core_first_global_page =
        ring_core_id * w0_w1_pages_per_ring_core_total + w0_w1_layer_offset_in_ring_core;
    const uint32_t w2_ring_core_first_global_page =
        ring_core_id * w2_pages_per_ring_core_total + w2_layer_offset_in_ring_core;

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];

        // Shared experts are TP-split on the intermediate dim and front-packed (real TpNt slice at
        // the front of each core's full-Nt shard, zeros after -- add_shared_expert_weights). Read
        // only the real prefix: W0/W1 layout is Nt-outer, so the prefix is a contiguous shortened
        // read. The compute kernel zero-fills the produced in2 gap so the full W2 walk stays correct.
        const bool is_shared_expert = expert_id >= num_experts - num_shared_experts;
        const uint32_t w0_w1_blocks_this_expert =
            is_shared_expert ? Cfg::w0_w1_blocks_per_shared_expert : Cfg::w0_w1_blocks_per_expert;

        // Per-expert slice's first GLOBAL page id (in the FULL flat tensor across all banks).
        const uint32_t w0_w1_slice_first_global_page =
            w0_w1_ring_core_first_global_page + expert_id * w0_w1_pages_per_logical_shard;
        const uint32_t w2_slice_first_global_page =
            w2_ring_core_first_global_page + expert_id * w2_pages_per_logical_shard;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            //-------------------------------------------------------------------------
            // Pipelined reading of W0/W1 -- bank-run loop
            //-------------------------------------------------------------------------
            // Walk the slice [slice_first_global_page, slice_first_global_page +
            // pages_per_logical_shard) page-by-page, batching reads within each bank. Each
            // block issues 2 transactions of `tiles_per_txn` contiguous tiles. The
            // static_asserts above guarantee bank boundaries land on txn boundaries, so we
            // never split a single transaction across two banks. We may re-set_state
            // mid-block though if the SECOND txn of a block lands in a different bank.
            //
            // shard_idx (= gp / pages_per_bank_total) is the placement-order index in
            // [0, num_banks); we translate to the chip bank id via shard_to_bank[].
            uint32_t w0_w1_global_page = w0_w1_slice_first_global_page;

            for (uint32_t block_id = 0; block_id < w0_w1_blocks_this_expert; ++block_id) {
                // Set trid (persists in NOC_PACKET_TAG cmd_buf; subsequent fast_reads inherit it).
                noc_async_read_set_trid(trid_to_issue);

                // Issue 2 transactions of `tiles_per_txn` (=14) tiles each.
                // First transaction:
                {
                    const uint32_t shard_idx = w0_w1_global_page / w0_w1_pages_per_bank_total;
                    const uint32_t in_bank_page = w0_w1_global_page - shard_idx * w0_w1_pages_per_bank_total;
                    const uint32_t in_bank_byte_offset = in_bank_page * w0_w1_tile_size + w0_w1_addr;
                    const uint32_t bank_id = shard_to_bank[shard_idx];
                    if (shard_idx != cur_shard_idx_w0) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w0_w1_bytes_per_txn, vchannel);
                        cur_shard_idx_w0 = shard_idx;
                    }
                    noc_async_read_one_packet_with_state_with_trid<
                        /*skip_ptr_update=*/false,
                        /*skip_cmdbuf_chk=*/true>(
                        get_noc_addr_from_bank_id<true>(bank_id, 0),
                        in_bank_byte_offset,
                        slot_addr[slot_to_issue],
                        trid_to_issue);
                    w0_w1_global_page += w0_w1_tiles_per_txn;
                }
                // Second transaction (may cross a bank boundary):
                {
                    const uint32_t shard_idx = w0_w1_global_page / w0_w1_pages_per_bank_total;
                    const uint32_t in_bank_page = w0_w1_global_page - shard_idx * w0_w1_pages_per_bank_total;
                    const uint32_t in_bank_byte_offset = in_bank_page * w0_w1_tile_size + w0_w1_addr;
                    const uint32_t bank_id = shard_to_bank[shard_idx];
                    if (shard_idx != cur_shard_idx_w0) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w0_w1_bytes_per_txn, vchannel);
                        cur_shard_idx_w0 = shard_idx;
                    }
                    noc_async_read_one_packet_with_state_with_trid<
                        /*skip_ptr_update=*/false,
                        /*skip_cmdbuf_chk=*/true>(
                        get_noc_addr_from_bank_id<true>(bank_id, 0),
                        in_bank_byte_offset,
                        slot_addr[slot_to_issue] + w0_w1_bytes_per_txn,
                        trid_to_issue);
                    w0_w1_global_page += w0_w1_tiles_per_txn;
                }

                ADVANCE_SLOT(slot_to_issue);
                ADVANCE_TRID(trid_to_issue);

                // Only when we first start the pipeline, we don't have any txns in flight
                if (txns_in_flight) {
                    noc_async_read_barrier_with_trid(trid_to_wait);
                    cb_r2c_w0_w1.push_back(w0_w1_tiles_per_block);

                    ADVANCE_TRID(trid_to_wait);

                    // Reserve for next block
                    cb_r2c_w0_w1.reserve_back(w0_w1_tiles_per_block * 2);
                }
                txns_in_flight = true;
            }

            //-------------------------------------------------------------------------
            // Pipelined reading of W2 -- bank-run loop
            //-------------------------------------------------------------------------
            uint32_t w2_global_page = w2_slice_first_global_page;

            // Read the FULL Nt-tall W2 for every expert, including shared experts. Shared-expert W2
            // is zero-padded to full Nt height (add_shared_expert_weights); the zero rows are inert
            // under the full contraction the compute kernel performs.
            for (uint32_t block_id = 0; block_id < Cfg::w2_blocks_per_expert; ++block_id) {
                noc_async_read_set_trid(trid_to_issue);

                // First transaction:
                {
                    const uint32_t shard_idx = w2_global_page / w2_pages_per_bank_total;
                    const uint32_t in_bank_page = w2_global_page - shard_idx * w2_pages_per_bank_total;
                    const uint32_t in_bank_byte_offset = in_bank_page * w2_tile_size + w2_addr;
                    const uint32_t bank_id = shard_to_bank[shard_idx];
                    if (shard_idx != cur_shard_idx_w2) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w2_bytes_per_txn, vchannel);
                        cur_shard_idx_w2 = shard_idx;
                    }
                    noc_async_read_one_packet_with_state_with_trid<
                        /*skip_ptr_update=*/false,
                        /*skip_cmdbuf_chk=*/true>(
                        get_noc_addr_from_bank_id<true>(bank_id, 0),
                        in_bank_byte_offset,
                        slot_addr[slot_to_issue],
                        trid_to_issue);
                    w2_global_page += w2_tiles_per_txn;
                }
                // Second transaction (may cross a bank boundary):
                {
                    const uint32_t shard_idx = w2_global_page / w2_pages_per_bank_total;
                    const uint32_t in_bank_page = w2_global_page - shard_idx * w2_pages_per_bank_total;
                    const uint32_t in_bank_byte_offset = in_bank_page * w2_tile_size + w2_addr;
                    const uint32_t bank_id = shard_to_bank[shard_idx];
                    if (shard_idx != cur_shard_idx_w2) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w2_bytes_per_txn, vchannel);
                        cur_shard_idx_w2 = shard_idx;
                    }
                    noc_async_read_one_packet_with_state_with_trid<
                        /*skip_ptr_update=*/false,
                        /*skip_cmdbuf_chk=*/true>(
                        get_noc_addr_from_bank_id<true>(bank_id, 0),
                        in_bank_byte_offset,
                        slot_addr[slot_to_issue] + w2_bytes_per_txn,
                        trid_to_issue);
                    w2_global_page += w2_tiles_per_txn;
                }

                ADVANCE_SLOT(slot_to_issue);
                ADVANCE_TRID(trid_to_issue);

                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_r2c_w2.push_back(w2_tiles_per_block);

                ADVANCE_TRID(trid_to_wait);

                // Reserve for next block
                cb_r2c_w2.reserve_back(w2_tiles_per_block * 2);
            }
        }
    }

    // Drain the pipeline - the last txn in flight
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_r2c_w2.push_back(w2_tiles_per_block);

    // We have one extra slot reserved, which we won't use.
    // For CB hygiene, we can push it back.
    cb_r2c_w2.push_back(w2_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
