// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "moe_ring_common.h"

// Weight CB slots (blocks): Cfg::weight_cb_slots -- 3 for 14- and 20-tile transactions (4 for 10-tile); all but one
// slot hold blocks whose DRAM reads are in flight.
#define NUM_SLOTS Cfg::weight_cb_slots

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
    // Per-shape DRAM transaction size of both weight streams (moe_ring::tiles_per_txn_for_shape: 14, or 10)
    constexpr uint32_t tiles_per_txn = get_named_compile_time_arg_val("tiles_per_txn");

    using Cfg = moe_ring::MoeRingConfig<Ht, Nt, num_cores, has_bias, shared_expert_tp_factor, tiles_per_txn>;

    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    // Number of physical DRAM banks the HEIGHT_SHARDED weight tensor lives on. The public
    // API requires ring N to equal the live bank count (12 on WH, 7/8 on BH), so each ring
    // core's slice is exactly one bank. The bank-run loop is retained for correctness when
    // direct prim callers pass a different ring size.
    constexpr uint32_t num_banks = get_named_compile_time_arg_val("num_banks");
    // W2: per-ring-core total tile-page count (across ALL layers and ALL experts). Derived from
    // the HEIGHT_SHARDED weight tensor's total page count divided by num_cores (the prepare
    // function emits a leading dim of num_cores; HEIGHT_SHARDED keeps the byte order so the
    // flat layout is core-major). W0/W1 uses the per-expert bank-balanced layout below instead.
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
    // The host appends `num_banks` entries here. The bank-run loops below derive a
    // shard_idx from the page position, then translate via this table to get the
    // actual chip bank to feed `get_noc_addr_from_bank_id`.
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
    // The transaction size is a per-shape parameter (Cfg::tiles_per_txn: 14, or 20 for the 2560/640 expert),
    // the same for both streams.
    constexpr uint32_t w0_w1_txns_per_block = Cfg::txns_per_block;
    constexpr uint32_t w0_w1_tiles_per_txn = Cfg::tiles_per_txn;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28 (10 * 2)

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = Cfg::txns_per_block;
    constexpr uint32_t w2_tiles_per_txn = Cfg::tiles_per_txn;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;  // 14 * 2 = 28 (10 * 2)

    //-------------------------------------------------------------------------
    // DRAM Reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_bytes_per_block = w0_w1_tiles_per_block * w0_w1_tile_size;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_w1_tile_size;
    [[maybe_unused]] constexpr uint32_t w2_bytes_per_block = w2_tiles_per_block * w2_tile_size;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w2_tile_size;

    // Bank-run loop invariant: w0_w1 and w2 share ONE physical NoC read cmd-buf (target bank
    // coordinates and size, set by noc_async_read_one_packet_set_state), tracked by the single
    // cur_shard_idx below (both tensors are sharded over the same shard_to_bank table). A core's
    // compact w0_w1 slice can end in the next bank shard while its w2 slice lives in its own,
    // so the streams must not keep separate "current shard" caches. As long as both streams
    // use the same bytes_per_txn a set_state for either stream is valid for both. If a future
    // config diverges these sizes, re-set the size at every stream boundary or move to
    // per-stream cmd-bufs.
    static_assert(
        w0_w1_bytes_per_txn == w2_bytes_per_txn,
        "Bank-run loop assumes w0_w1 and w2 share identical bytes_per_txn (NoC cmd-buf size).");

    // W0/W1 compact layout (moe_ring_common.h, MoeRingConfig): ring core r stores only its
    // shard_tiles(Nt, r) gate/up columns, so the cores' per-expert slices differ in size. For every
    // (layer, expert) the cores' slices are laid back to back (core r at block offset
    // w0_w1_block_offset_lut[r]) and that stream, zero-padded to num_banks * bank_pages_per_expert
    // pages, is cut into num_banks equal pieces: piece b of expert (l, e) sits in bank shard b at
    //     in_bank_page = (l * num_experts + e) * bank_pages_per_expert + (stream page - b * bank_pages_per_expert)
    // Every bank thus holds the same bytes per expert; a core whose slice crosses a piece boundary
    // reads its tail from the next shard. With equal slices and num_cores == num_banks this is the
    // plain core-major layout (piece r == core r's slice).
    //
    // `shard_idx` is the placement-order index in [0, num_banks); the chip bank id is
    // obtained via `shard_to_bank[shard_idx]` (host computes this from the actual
    // buffer placement returned by `buffer()->get_buffer_page_mapping()`).
    constexpr auto w0_w1_block_offset_lut = moe_ring::make_w0_w1_block_offset_lut<Cfg, num_cores>();
    constexpr uint32_t w0_w1_bank_pages_per_expert =
        Cfg::w0_w1_bank_blocks_per_expert(num_banks) * w0_w1_tiles_per_block;
    // Each transaction is `tiles_per_txn` (14 or 20) contiguous tiles. For the bank-run to work
    // without splitting a single transaction across a bank boundary, the slice offsets and the
    // bank piece size are whole blocks (multiples of the transaction tile count), and the
    // cores' slices must fit in the num_banks pieces of one expert.
    static_assert(
        w0_w1_bank_pages_per_expert % w0_w1_tiles_per_txn == 0,
        "w0_w1 bank_pages_per_expert must be a multiple of tiles_per_txn");
    static_assert(
        w0_w1_block_offset_lut[num_cores] <= num_banks * Cfg::w0_w1_bank_blocks_per_expert(num_banks),
        "the cores' w0_w1 slices of one expert must fit in its num_banks bank pieces");

    constexpr uint32_t w2_pages_per_logical_shard = Cfg::w2_blocks_per_expert * w2_tiles_per_block;
    constexpr uint32_t w2_pages_total = num_cores * w2_pages_per_ring_core_total;
    static_assert(w2_pages_total % num_banks == 0, "w2 pages_total must be divisible by num_banks");
    constexpr uint32_t w2_pages_per_bank_total = w2_pages_total / num_banks;
    static_assert(
        w2_pages_per_logical_shard % w2_tiles_per_txn == 0,
        "w2 pages_per_logical_shard must be a multiple of tiles_per_txn");
    static_assert(
        w2_pages_per_bank_total % w2_tiles_per_txn == 0, "w2 pages_per_bank_total must be a multiple of tiles_per_txn");

    // Layer's per-ring-core (W2) / per-bank (W0/W1) stride in pages.
    constexpr uint32_t w2_layer_pages_per_ring_core = num_experts * w2_pages_per_logical_shard;
    constexpr uint32_t w0_w1_layer_offset_in_bank = layer_id * num_experts * w0_w1_bank_pages_per_expert;
    constexpr uint32_t w2_layer_offset_in_ring_core = layer_id * w2_layer_pages_per_ring_core;

    //-------------------------------------------------------------------------
    // CB addresses
    //-------------------------------------------------------------------------
    const uint32_t w_cb_base_addr = cb_r2c_w0_w1.get_write_ptr();

    // Precompute slot addresses (avoid multiply in hot loop)
    // Each slot holds 2 transactions (one block)
    uint32_t slot_addr[NUM_SLOTS];
    for (uint32_t slot = 0; slot < NUM_SLOTS; ++slot) {
        slot_addr[slot] = w_cb_base_addr + slot * w0_w1_bytes_per_block;
    }

    //-------------------------------------------------------------------------
    // Variables to track pipeline state
    //-------------------------------------------------------------------------
    // Up to blocks_in_flight blocks have DRAM reads outstanding (one trid each); the remaining slot is the
    // block compute is consuming. With 3 slots this is the original issue-one / wait-for-the-previous pipeline.
    constexpr uint32_t blocks_in_flight = NUM_SLOTS - 1;
    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot_to_issue = 0;
    uint32_t blocks_pending = 0;

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

    // We reserve the blocks issued before the first wait to kick start the pipeline, and then it is steady state
    cb_r2c_w0_w1.reserve_back(w0_w1_tiles_per_block * (blocks_in_flight - 1));

    // Pre-set state for this ring core's first bank (WH fast path: when
    // pages_per_ring_core_total <= pages_per_bank_total). The bank-run loop below will
    // re-set_state only when shard_idx changes.
    // Device 2.0 migration: legacy primitives retained: noc_async_read_set_trid /
    // noc_async_read_one_packet_set_state / noc_async_read_one_packet_with_state_with_trid /
    // noc_async_read_barrier_with_trid are the trid-pipelined state-machine API used to
    // drive a triple-buffered DRAM read pipeline; Device 2.0 Noc wrapper does not yet expose
    // typed equivalents for the set_state / with_state / with_trid family
    // This ring core's slice position inside every (layer, expert) stream: the bank piece it starts in and
    // the page offset inside that piece.
    const uint32_t w0_w1_core_first_stream_page = w0_w1_block_offset_lut[ring_core_id] * w0_w1_tiles_per_block;
    const uint32_t w0_w1_core_first_shard_idx = w0_w1_core_first_stream_page / w0_w1_bank_pages_per_expert;
    const uint32_t w0_w1_core_first_piece_page =
        w0_w1_core_first_stream_page - w0_w1_core_first_shard_idx * w0_w1_bank_pages_per_expert;
    const uint32_t initial_shard_idx_w0 = w0_w1_core_first_shard_idx;
    const uint32_t initial_bank_id_w0 = shard_to_bank[initial_shard_idx_w0];
    {
        const uint64_t initial_dram_noc_addr_w0 = get_noc_addr_from_bank_id<true>(initial_bank_id_w0, 0);
        noc_async_read_one_packet_set_state<true>(initial_dram_noc_addr_w0, w0_w1_bytes_per_txn, vchannel);
    }
    // Track the currently set_state'd bank shard of the (shared) read cmd-buf, for w0_w1 and w2
    // reads alike. Initially equal to the bank we just set above. The bank-run loops only
    // re-set_state when the shard of the next transaction differs.
    uint32_t cur_shard_idx = initial_shard_idx_w0;

    // This ring core's first W2 global page id for the CURRENT layer.
    const uint32_t w2_ring_core_first_global_page =
        ring_core_id * w2_pages_per_ring_core_total + w2_layer_offset_in_ring_core;

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];

        // Shared experts are TP-split on the intermediate dim and front-packed (real TpNt slice at
        // the front of each core's full-Nt shard, zeros after -- add_shared_expert_weights). Read
        // only the real prefix: W0/W1 layout is Nt-outer, so the prefix is a contiguous shortened
        // read. The compute kernel zero-fills the produced in2 gap so the full W2 walk stays correct.
        // Routed experts read the core's whole compact slice (its logical columns only).
        const bool is_shared_expert = expert_id >= num_experts - num_shared_experts;
        const uint32_t w0_w1_blocks_this_expert = moe_ring::w0_w1_blocks_for_cols(
            Cfg::w0_w1_prod_cols(ring_core_id, is_shared_expert),
            Cfg::w0_w1_blocks_per_col,
            Cfg::w0_w1_blocks_per_half_col);

        // This expert's first page in every bank shard.
        const uint32_t w0_w1_expert_first_bank_page =
            w0_w1_layer_offset_in_bank + expert_id * w0_w1_bank_pages_per_expert;
        const uint32_t w2_slice_first_global_page =
            w2_ring_core_first_global_page + expert_id * w2_pages_per_logical_shard;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            //-------------------------------------------------------------------------
            // Pipelined reading of W0/W1 -- bank-run loop
            //-------------------------------------------------------------------------
            // Walk this core's slice of the expert stream txn by txn, batching reads within each
            // bank piece. Each block issues 2 transactions of `tiles_per_txn` contiguous tiles.
            // The static_asserts above guarantee piece boundaries land on txn boundaries, so we
            // never split a single transaction across two banks. We may re-set_state
            // mid-block though if the SECOND txn of a block lands in the next piece.
            //
            // shard_idx is the placement-order index in [0, num_banks); we translate to the chip
            // bank id via shard_to_bank[].
            uint32_t w0_w1_shard_idx = w0_w1_core_first_shard_idx;
            uint32_t w0_w1_piece_page = w0_w1_core_first_piece_page;

            for (uint32_t block_id = 0; block_id < w0_w1_blocks_this_expert; ++block_id) {
                // Set trid (persists in NOC_PACKET_TAG cmd_buf; subsequent fast_reads inherit it).
                noc_async_read_set_trid(trid_to_issue);

                // Issue 2 transactions of `tiles_per_txn` tiles each.
                // First transaction:
                {
                    const uint32_t shard_idx = w0_w1_shard_idx;
                    const uint32_t in_bank_page = w0_w1_expert_first_bank_page + w0_w1_piece_page;
                    const uint32_t in_bank_byte_offset = in_bank_page * w0_w1_tile_size + w0_w1_addr;
                    const uint32_t bank_id = shard_to_bank[shard_idx];
                    if (shard_idx != cur_shard_idx) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w0_w1_bytes_per_txn, vchannel);
                        cur_shard_idx = shard_idx;
                    }
                    noc_async_read_one_packet_with_state_with_trid<
                        /*skip_ptr_update=*/false,
                        /*skip_cmdbuf_chk=*/true>(
                        get_noc_addr_from_bank_id<true>(bank_id, 0),
                        in_bank_byte_offset,
                        slot_addr[slot_to_issue],
                        trid_to_issue);
                    w0_w1_piece_page += w0_w1_tiles_per_txn;
                    if (w0_w1_piece_page == w0_w1_bank_pages_per_expert) {
                        ++w0_w1_shard_idx;
                        w0_w1_piece_page = 0;
                    }
                }
                // Second transaction (may cross a bank boundary):
                {
                    const uint32_t shard_idx = w0_w1_shard_idx;
                    const uint32_t in_bank_page = w0_w1_expert_first_bank_page + w0_w1_piece_page;
                    const uint32_t in_bank_byte_offset = in_bank_page * w0_w1_tile_size + w0_w1_addr;
                    const uint32_t bank_id = shard_to_bank[shard_idx];
                    if (shard_idx != cur_shard_idx) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w0_w1_bytes_per_txn, vchannel);
                        cur_shard_idx = shard_idx;
                    }
                    noc_async_read_one_packet_with_state_with_trid<
                        /*skip_ptr_update=*/false,
                        /*skip_cmdbuf_chk=*/true>(
                        get_noc_addr_from_bank_id<true>(bank_id, 0),
                        in_bank_byte_offset,
                        slot_addr[slot_to_issue] + w0_w1_bytes_per_txn,
                        trid_to_issue);
                    w0_w1_piece_page += w0_w1_tiles_per_txn;
                    if (w0_w1_piece_page == w0_w1_bank_pages_per_expert) {
                        ++w0_w1_shard_idx;
                        w0_w1_piece_page = 0;
                    }
                }

                ADVANCE_SLOT(slot_to_issue);
                ADVANCE_TRID(trid_to_issue);

                // While the pipeline fills (the first blocks_in_flight - 1 blocks) nothing is waited on
                if (++blocks_pending == blocks_in_flight) {
                    noc_async_read_barrier_with_trid(trid_to_wait);
                    cb_r2c_w0_w1.push_back(w0_w1_tiles_per_block);

                    ADVANCE_TRID(trid_to_wait);
                    --blocks_pending;

                    // Reserve for next block (the blocks in flight and the next one)
                    cb_r2c_w0_w1.reserve_back(w0_w1_tiles_per_block * blocks_in_flight);
                }
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
                    if (shard_idx != cur_shard_idx) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w2_bytes_per_txn, vchannel);
                        cur_shard_idx = shard_idx;
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
                    if (shard_idx != cur_shard_idx) {
                        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
                        noc_async_read_one_packet_set_state<true>(bank_base, w2_bytes_per_txn, vchannel);
                        cur_shard_idx = shard_idx;
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

                if (++blocks_pending == blocks_in_flight) {
                    noc_async_read_barrier_with_trid(trid_to_wait);
                    cb_r2c_w2.push_back(w2_tiles_per_block);

                    ADVANCE_TRID(trid_to_wait);
                    --blocks_pending;

                    // Reserve for next block (the blocks in flight and the next one)
                    cb_r2c_w2.reserve_back(w2_tiles_per_block * blocks_in_flight);
                }
            }
        }
    }

    // Drain the pipeline - the blocks still in flight
    for (; blocks_pending > 0; --blocks_pending) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_r2c_w2.push_back(w2_tiles_per_block);
        ADVANCE_TRID(trid_to_wait);
    }

    // We have one extra slot reserved, which we won't use.
    // For CB hygiene, we can push it back.
    cb_r2c_w2.push_back(w2_tiles_per_block);
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
