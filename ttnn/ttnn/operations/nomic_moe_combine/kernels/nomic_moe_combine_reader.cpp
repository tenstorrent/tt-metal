// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_combine — reader of choice k (two instances: k = 0 on RISCV_0 / NOC_0, k = 1 on RISCV_1 / NOC_1).
//
// Block schedule (op_design.md § Block schedule), per core, one block = the whole output shard:
//   load_index_block       one DRAM read: this shard's core_row_tiles x 32 row ids of choice k (contiguous)
//   prefetch_weights_block faces 0 and 2 (the only ones the column broadcast reads) of the shard's weight tiles
//   build_bias_tiles       reader 0 only: 2W reads of 64 B of the fp32 bias row straight into row 0 of faces 0 / 1
//                          of each of the W bias tiles
//   gather_rows_block      per tile row: 32 one-packet reads of W*64 B (the core's hidden slice of each chosen
//                          expert row) straight into the row-major rows block, in batches of batch_reads under
//                          batch_depth rotating transaction ids; W pages pushed when the tile row's last batch lands
//
// The index, weight and bias reads go out first (TensorAccessor, trid-tagged), each waited by its own transaction id
// only with later reads behind it (MOE_GLUE §4: an id barrier can pass before the most recent read under that id left
// the command buffer). The row reads use the NoC's stateful one-packet path: size and command fixed once, per read
// only the source coordinate + address and the destination are written. The final drain is the full read barrier.
//
// group_shared_index_weights (Refinement 2): the G owners of a row group (cores i .. i+G-1 of the shard order, member
// g = hidden slice) need the same index slice and weight faces. With share_index / share_weights, only choice k's
// leader (member leader_member) reads them from DRAM; as each lands it unicasts the bytes into the same L1 offsets of
// the peers' CBs (all CBs sit at the same address on every core), and once those writes are acked bumps each peer's
// semaphore of choice k by the items delivered (the index = 1 item if shared, then one per weight tile, in order).
// A peer's reader k waits on that count instead of its own DRAM reads.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#ifdef ABLATE_ZONES
#include "api/debug/kernel_profiler.hpp"
#define NMC_ZONE(name) DeviceZoneScopedN(name)
#else
#define NMC_ZONE(name)
#endif

namespace {

constexpr uint32_t TILE_DIM = 32;

}  // namespace

void kernel_main() {
    constexpr uint32_t choice = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_raw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t shard_col_tiles = get_compile_time_arg_val(5);       // W
    constexpr uint32_t shard_row_tiles = get_compile_time_arg_val(6);       // r
    constexpr uint32_t rows_depth_tile_rows = get_compile_time_arg_val(7);  // rows CB depth knob
    constexpr uint32_t rows_page_bytes = get_compile_time_arg_val(8);       // bank stride of one expert row
    constexpr uint32_t batch_reads = get_compile_time_arg_val(9);
    constexpr uint32_t batch_depth = get_compile_time_arg_val(10);
    constexpr uint32_t token_tiles = get_compile_time_arg_val(11);  // T' / 32
    // 1: the index read goes out alone and is waited with the full barrier before the weight / bias reads are issued,
    // so no DRAM request of this core queues ahead of another core's index read on the index's bank.
    constexpr bool index_first = get_compile_time_arg_val(12) != 0;
    // Weight tiles issued ahead of the row reads: 1..shard_row_tiles (shard_row_tiles = all at start-up).
    constexpr uint32_t weights_lookahead = get_compile_time_arg_val(13);
    static_assert(weights_lookahead >= 1, "tile row 0's weights go out at start-up");
    // Request VC of the side reads (weights, bias); the row reads keep the default read VC.
    constexpr uint32_t side_read_vc = get_compile_time_arg_val(14);
    constexpr uint32_t row_read_vc = 1;  // the firmware's default static read VC
    // 1: each weight tile's faces 0-2 in one 3 KB request; 0: faces 0 and 2 as two 1 KB requests.
    constexpr bool weights_one_read = get_compile_time_arg_val(15) != 0;
    // Row batches issued before the side (weight / bias) DRAM reads: 0 = side reads at start-up (op_design). With
    // > 0 the first tile row's row reads get ahead of the grid's DRAM burst, and the index is waited alone.
    constexpr uint32_t side_after_batches = get_compile_time_arg_val(16);
    // group_shared_index_weights: hidden_split = G (members of a row group); sharing is a no-op at G = 1.
    constexpr uint32_t hidden_split = get_compile_time_arg_val(17);
    constexpr bool share_index = get_compile_time_arg_val(18) != 0 && hidden_split > 1;
    constexpr bool share_weights = get_compile_time_arg_val(19) != 0 && hidden_split > 1;
    constexpr uint32_t delivered_semaphore_id = get_compile_time_arg_val(20);  // choice k's own semaphore
    constexpr uint32_t leader_member = get_compile_time_arg_val(21);           // the member that reads choice k's
    static_assert(leader_member < hidden_split, "the leader is a member of the group");
    constexpr uint32_t num_peers = hidden_split - 1;
    // Core tables (Refinement 4): the core of logical (x, y) is core i = y * grid_x + x of the shard order (the host
    // refuses a shard grid where it is not); its per-core values are the record i of the common args.
    constexpr uint32_t grid_x = get_compile_time_arg_val(22);
    constexpr uint32_t num_cores = get_compile_time_arg_val(23);
    constexpr uint32_t record_field_bits = get_compile_time_arg_val(24);
    constexpr auto index_args = TensorAccessorArgs<25>();
    constexpr auto weights_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    // No per-core runtime args (each costs the host ~0.3 us of dispatch a core). Common runtime args:
    //   0 .. 3                       the four input buffer addresses (the only per-call values)
    //   records_crt + i              core i's record: tr0 | core_row_tiles | g | bias copy, record_field_bits each
    //   noc_xy_crt + i               core i's NoC coordinate: x | y << 16 (the leader's peers are cores of its group)
    const uint32_t rows_address = get_common_arg_val<uint32_t>(0);
    const uint32_t index_address = get_common_arg_val<uint32_t>(1);
    const uint32_t weights_address = get_common_arg_val<uint32_t>(2);
    const uint32_t bias_address = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t records_crt = 4;
    constexpr uint32_t noc_xy_crt = records_crt + num_cores;
    constexpr uint32_t field_mask = (1u << record_field_bits) - 1;
    const uint32_t core_index = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t record = get_common_arg_val<uint32_t>(records_crt + core_index);
    const uint32_t first_tile_row = record & field_mask;  // tr0: flat tile row of the shard's first row
    const uint32_t core_row_tiles = (record >> record_field_bits) & field_mask;
    const uint32_t hidden_slice = (record >> (2 * record_field_bits)) & field_mask;  // g
    const uint32_t bias_copy = (record >> (3 * record_field_bits)) & field_mask;
    // Peer p of the group (its members other than leader_member): core group_first + member.
    const uint32_t group_first = core_index - hidden_slice;
    const auto peer_noc_xy = [&](uint32_t p) {
        const uint32_t member = p < leader_member ? p : p + 1;
        return get_common_arg_val<uint32_t>(noc_xy_crt + group_first + member);
    };
    const bool leader = hidden_slice == leader_member;
    const bool index_from_leader = share_index && !leader;
    const bool weights_from_leader = share_weights && !leader;
    const bool forwards = leader && (share_index || share_weights);
    const uint32_t delivered_semaphore = get_semaphore(delivered_semaphore_id);
    auto* delivered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(delivered_semaphore);
    // Items the leader delivers before weight tile 0: the index slice, when shared.
    constexpr uint32_t weights_item_base = share_index ? 1 : 0;

    static_assert(batch_depth >= 2, "a batch is waited for by id only with a later batch behind it");
    static_assert(TILE_DIM % batch_reads == 0, "a tile row's reads must end on a batch boundary");
    // Ids 1..batch_depth: row batches; then the index, the bias, and one id per tile row's weight tile (so each tile
    // row's weights are pushed as soon as they land and compute starts tile row 0 without waiting for the rest).
    constexpr uint32_t index_trid = batch_depth + 1;
    constexpr uint32_t bias_trid = batch_depth + 2;        // the bias slice's one DRAM read into staging
    constexpr uint32_t bias_place_trid = batch_depth + 3;  // its 2W local placement reads
    constexpr uint32_t first_weights_trid = batch_depth + 4;
    static_assert(first_weights_trid + shard_row_tiles - 1 <= 15, "transaction ids are 4-bit");

    constexpr uint32_t segment_bytes = shard_col_tiles * TILE_DIM * sizeof(uint16_t);  // W*64 B of one bf16 row
    static_assert(segment_bytes <= NOC_MAX_BURST_SIZE, "a row segment must be one NoC packet");
    constexpr uint32_t tile_row_bytes = TILE_DIM * segment_bytes;  // 32 rows x W*32 values = W bf16 pages
    constexpr uint32_t index_tile_row_bytes = TILE_DIM * sizeof(uint32_t);
    constexpr uint32_t weight_tile_bytes = get_tile_size(cb_weights_raw);
    constexpr uint32_t weight_face_bytes = weight_tile_bytes / 4;
    constexpr uint32_t bias_tile_bytes = get_tile_size(cb_bias_tiles);
    constexpr uint32_t bias_face_bytes = bias_tile_bytes / 4;
    constexpr uint32_t bias_half_row_bytes = TILE_DIM / 2 * sizeof(float);  // 16 fp32 = 64 B (one face row)
    constexpr uint32_t bias_slice_bytes = shard_col_tiles * TILE_DIM * sizeof(float);
    constexpr uint32_t bias_staging_offset = 2 * bias_face_bytes;  // faces 2-3 of bias tile 0 (never read by compute)
    static_assert(bias_slice_bytes <= bias_tile_bytes - bias_staging_offset, "bias staging overruns bias tile 0");

    const Noc noc;
    const auto index_accessor = TensorAccessor(index_args, index_address);
    const auto weights_accessor = TensorAccessor(weights_args, weights_address);
    const auto bias_accessor = TensorAccessor(bias_args, bias_address);

    // ---- load_index_block: one read of core_row_tiles * 128 B of page `choice` ----
    cb_reserve_back(cb_index, core_row_tiles);
    const uint32_t index_l1 = get_write_ptr(cb_index);
    const uint32_t index_bytes = core_row_tiles * index_tile_row_bytes;
    if (!index_from_leader) {
        noc.async_read<NocOptions::TXN_ID>(
            index_accessor,
            CoreLocalMem<uint32_t>(index_l1),
            core_row_tiles * index_tile_row_bytes,
#ifdef ABLATE_SAMEINDEX
            {.page_id = 0, .offset_bytes = first_tile_row * index_tile_row_bytes},
#else
            {.page_id = choice, .offset_bytes = first_tile_row * index_tile_row_bytes},
#endif
            {},
            {.trid = index_trid});
    }
    // A peer that takes its weights from the leader has no weight reads to put behind its index read.
    const bool index_alone = index_first || side_after_batches > 0 || weights_from_leader;
    if (index_alone && !index_from_leader) {
        noc_async_read_barrier();  // no later read behind it: the full barrier (an id barrier would be racy)
    }

    // ---- leader: forward landed index / weight bytes to the peers' same L1 offsets; signal once acked ----
    uint32_t items_forwarded = 0;  // items whose writes are issued
    uint32_t items_signalled = 0;  // items the peers' semaphores count
    const auto forward_to_peers = [&](uint32_t l1, uint32_t bytes) {
        for (uint32_t p = 0; p < num_peers; ++p) {
            const uint32_t xy = peer_noc_xy(p);
            const uint32_t x = xy & 0xFFFF;
            const uint32_t y = xy >> 16;
            noc_async_write(l1, get_noc_addr(x, y, l1), bytes);
        }
    };
    // Bumps the peers' semaphores by the items forwarded since the last signal, once every write issued so far is
    // acked (non-blocking unless block: the writes must have landed before a peer can see the count).
    const auto signal_peers = [&](bool block) {
        if (items_signalled == items_forwarded) {
            return;
        }
        if (block) {
            noc_async_write_barrier();
        } else if (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            return;
        }
        for (uint32_t p = 0; p < num_peers; ++p) {
            const uint32_t xy = peer_noc_xy(p);
            const uint32_t x = xy & 0xFFFF;
            const uint32_t y = xy >> 16;
            noc_semaphore_inc(get_noc_addr(x, y, delivered_semaphore), items_forwarded - items_signalled);
        }
        items_signalled = items_forwarded;
    };

    // Weight faces and bias pieces come in a few reads from one DRAM page each: the page's NoC address is computed
    // once (TensorAccessor) and its pieces go out on the stateful one-packet path (coordinate + size fixed per page,
    // per piece only the address and destination). Per-piece TensorAccessor reads cost ~130 cycles of issue each,
    // which put reader 0 (index + weights + 2W bias pieces) 2-3 us behind reader 1.
    const auto issue_page_pieces = [&](uint64_t page_noc_address, uint32_t piece_bytes, uint32_t trid) {
        noc_async_read_one_packet_set_state<true>(page_noc_address, piece_bytes, side_read_vc);
        noc_async_read_set_trid(trid);
    };
    const auto issue_piece = [&](uint64_t page_noc_address, uint32_t offset_bytes, uint32_t dst_l1) {
        noc_async_read_one_packet_with_state((uint32_t)page_noc_address + offset_bytes, dst_l1);
    };

    // ---- prefetch_weights_block: faces 0 and 2 of each weight tile, one id per tile row. The first
    // weights_lookahead tile rows' tiles are issued here, behind the index read; tile row t + weights_lookahead's tile
    // is issued as tile row t's row reads start, so the grid's DRAM requests are spread over the gather instead of one
    // burst at start-up (that burst backs up behind DRAM and stalls the L1 row reads sharing its NoC paths). ----
    cb_reserve_back(cb_weights_raw, core_row_tiles);
    const uint32_t weights_l1 = get_write_ptr(cb_weights_raw);
    uint32_t weights_issued = 0;  // weight tiles whose DRAM reads are issued
    uint32_t weights_safe = 0;    // ... of which a row batch went out behind (their id checks are not racy)
    const auto issue_weight_tile = [&](uint32_t t) {
        if (weights_from_leader) {
            return;  // the leader delivers this tile
        }
        ++weights_issued;
        const uint64_t page_noc_address =
            weights_accessor.get_noc_addr(choice * token_tiles + first_tile_row + t, 0, noc_index);
#ifndef ABLATE_WEIGHTSREAD
        if constexpr (weights_one_read) {
            // Faces 0-2 in one request (face 1 is read but never used): half the DRAM requests, 1.5x the bytes.
            issue_page_pieces(page_noc_address, 3 * weight_face_bytes, first_weights_trid + t);
            issue_piece(page_noc_address, 0, weights_l1 + t * weight_tile_bytes);
        } else {
            issue_page_pieces(page_noc_address, weight_face_bytes, first_weights_trid + t);
            for (uint32_t face = 0; face < 4; face += 2) {
                issue_piece(
                    page_noc_address,
                    face * weight_face_bytes,
                    weights_l1 + t * weight_tile_bytes + face * weight_face_bytes);
            }
        }
#else
        issue_page_pieces(page_noc_address, weight_face_bytes, first_weights_trid + t);
#endif
    };

    // ---- build_bias_tiles (issue), reader 0 only: this core's W*32 fp32 bias values in ONE DRAM read, staged in
    // faces 2-3 of bias tile 0; once landed, 2W local NoC reads of 64 B place columns 32j .. 32j + 15 in row 0 of
    // face 0 and 32j + 16 .. 32j + 31 in row 0 of face 1 of tile j (the only rows the row-broadcast unpack reads).
    // One DRAM request a core: 2W 64 B DRAM requests a core (1536 over the grid at 8x512) measured ~3 us on this
    // reader (they back up at DRAM and stall the row reads); a RISC copy of the W*32 words measured 2-3.5 us. ----
    uint32_t bias_l1 = 0;
    if constexpr (choice == 0) {
        cb_reserve_back(cb_bias_tiles, shard_col_tiles);
        bias_l1 = get_write_ptr(cb_bias_tiles);
    }
    bool side_started = false;
    uint32_t side_issued_at = 0;  // row batches issued when the side reads went out
    const auto issue_side_reads = [&]() {
        side_started = true;
        const uint32_t startup_weight_tiles = core_row_tiles < weights_lookahead ? core_row_tiles : weights_lookahead;
        for (uint32_t t = 0; t < startup_weight_tiles; ++t) {
            issue_weight_tile(t);
        }
        if constexpr (choice == 0) {
#ifndef ABLATE_BIASREAD
            const uint64_t page_noc_address =
                bias_accessor.get_noc_addr(bias_copy, hidden_slice * bias_slice_bytes, noc_index);
            issue_page_pieces(page_noc_address, bias_slice_bytes, bias_trid);
            issue_piece(page_noc_address, 0, bias_l1 + bias_staging_offset);
#endif
        }
    };
    if constexpr (side_after_batches == 0) {
        issue_side_reads();
    }

    if (index_from_leader) {
        NMC_ZONE("NMC-IDXWAIT");
        noc_semaphore_wait_min(delivered, 1);  // invalidates the L1 cache: the ids below are fresh
    } else if (!index_alone) {
        NMC_ZONE("NMC-IDXWAIT");
        // The index read has the weight reads behind it: its id barrier is safe.
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = index_trid});
    }
    if (forwards && share_index) {
        // The peers' row reads wait on these ids: forward and signal at once (a NoC round trip on the leader).
        forward_to_peers(index_l1, index_bytes);
        ++items_forwarded;
        signal_peers(true);
    }
    auto* row_ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1);

    // ---- gather_rows_block ----
    // Stateful one-packet reads: the size (and command) is fixed here once; every TensorAccessor read above is
    // already issued, so nothing reprograms the read command buffer after this point.
    const uint32_t rows_base_l1 = get_write_ptr(cb_rows);
    const uint32_t source_base = rows_address + hidden_slice * segment_bytes;
    noc_async_read_one_packet_set_state<true>(
        get_noc_addr_from_bank_id<false>(0, source_base), segment_bytes, row_read_vc);

    const auto issue_row = [&](uint32_t row, uint32_t dst_l1) {
        // Interleaved L1: row r is page r on bank r % NUM_L1_BANKS, at (r / NUM_L1_BANKS) * page bytes.
        const uint32_t bank = row % NUM_L1_BANKS;
        const uint32_t address = source_base + (row / NUM_L1_BANKS) * rows_page_bytes;
        const uint64_t noc_address = get_noc_addr_from_bank_id<false>(bank, address, noc_index);
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf));
        NOC_CMD_BUF_WRITE_REG(
            noc_index,
            read_cmd_buf,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(noc_address >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        ncrisc_noc_read_with_state<noc_mode, true /*inc_num_issued*/, true /*one_packet*/, true /*skip_cmdbuf_chk*/>(
            noc_index, read_cmd_buf, (uint32_t)noc_address, dst_l1);
    };

    // Ring of batch_depth transaction ids. completes[slot] = 1 + the tile row whose reads end with that batch, or 0.
    uint32_t completes[batch_depth] = {};
    uint32_t issued = 0;
    uint32_t retired = 0;
    uint32_t landed_rows = 0;  // tile rows whose reads are all in
    uint32_t pushed_rows = 0;
    const auto mark_retired = [&]() {
        const uint32_t slot = retired % batch_depth;
        ++retired;
        if (completes[slot] != 0) {
            landed_rows = completes[slot];
            completes[slot] = 0;
        }
    };
    const auto push_landed = [&]() {
        for (; pushed_rows < landed_rows; ++pushed_rows) {
            NMC_ZONE("NMC-PUSH");
            cb_push_back(cb_rows, shard_col_tiles);
        }
    };

    uint32_t weights_pushed = 0;  // weight tiles (tile rows) pushed, in order
    // Reader 0's bias: 0 = staging read in flight, 1 = placement reads in flight, 2 = pushed.
    uint32_t bias_stage = choice == 0 ? 0 : 2;
#ifdef ABLATE_BIASREAD
    bias_stage = choice == 0 ? 1 : 2;
#endif
    const auto place_bias = [&]() {
        const uint64_t staging = get_noc_addr(my_x[noc_index], my_y[noc_index], bias_l1 + bias_staging_offset);
        issue_page_pieces(staging, bias_half_row_bytes, bias_place_trid);
        for (uint32_t j = 0; j < shard_col_tiles; ++j) {
            for (uint32_t half = 0; half < 2; ++half) {
                issue_piece(
                    staging,
                    (2 * j + half) * bias_half_row_bytes,
                    bias_l1 + j * bias_tile_bytes + half * bias_face_bytes);
            }
        }
        // Restore the row reads' one-packet size (the coordinate is rewritten per row read).
        noc_async_read_one_packet_set_state<true>(
            get_noc_addr_from_bank_id<false>(0, source_base), segment_bytes, row_read_vc);
    };
    // Weight id checks only for tiles with a row batch behind them, bias id checks only right after a row batch went
    // out (id checks are then not racy, MOE_GLUE §4); with block = true only after the final full barrier. light =
    // the weights and the group forwarding only (polled while waiting on row reads, so a peer's semaphore is bumped
    // as soon as the leader's writes are acked, not at the leader's next batch boundary).
    const auto poll_side_reads = [&](bool block, bool light) {
        if (weights_from_leader) {
            // Delivered by the leader: its semaphore counts the landed items (no NoC id check, never racy).
            if (block) {
                noc_semaphore_wait_min(delivered, weights_item_base + core_row_tiles);
            } else {
                invalidate_l1_cache();
            }
            const uint32_t landed = *delivered - weights_item_base;
            for (; weights_pushed < core_row_tiles && weights_pushed < landed; ++weights_pushed) {
                cb_push_back(cb_weights_raw, 1);
            }
        }
        while (!weights_from_leader && weights_pushed < core_row_tiles &&
               (block || (weights_pushed < weights_safe && ncrisc_noc_read_with_transaction_id_flushed(
                                                               noc_index, first_weights_trid + weights_pushed)))) {
            if (forwards && share_weights) {
                const uint32_t tile_l1 = weights_l1 + weights_pushed * weight_tile_bytes;
                for (uint32_t face = 0; face < 4; face += 2) {  // faces 0 and 2: all the column broadcast reads
                    forward_to_peers(tile_l1 + face * weight_face_bytes, weight_face_bytes);
                }
                ++items_forwarded;
            }
            ++weights_pushed;
            cb_push_back(cb_weights_raw, 1);
        }
        if (forwards) {
            signal_peers(block);
        }
        if (light || (!block && (!side_started || issued == side_issued_at))) {
            return;  // no row batch right behind the bias reads: their id checks would be racy
        }
        if constexpr (choice == 0) {
#ifndef ABLATE_BIASREAD
            if (bias_stage == 0 && (block || ncrisc_noc_read_with_transaction_id_flushed(noc_index, bias_trid))) {
                bias_stage = 1;
                invalidate_l1_cache();
                place_bias();
                if (block) {
                    noc_async_read_barrier();
                }
                return;  // the placement reads have no later read behind them yet
            }
#endif
            if (bias_stage == 1 && (block || ncrisc_noc_read_with_transaction_id_flushed(noc_index, bias_place_trid))) {
                bias_stage = 2;
                cb_push_back(cb_bias_tiles, shard_col_tiles);
            }
        }
    };

    // The oldest batch in flight; only ever called with a later batch behind it. Services the weights / forwarding
    // while it waits.
    const auto retire_next = [&]() {
        const uint32_t trid = retired % batch_depth + 1;
        while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
            poll_side_reads(false, true);
        }
        invalidate_l1_cache();
        mark_retired();
        push_landed();
    };

    {
        NMC_ZONE("NMC-ROWS");
        for (uint32_t row_idx = 0; row_idx < core_row_tiles; ++row_idx) {
            if (side_started && row_idx != 0 && row_idx + weights_lookahead - 1 < core_row_tiles) {
                issue_weight_tile(row_idx + weights_lookahead - 1);
                // Restore the row reads' one-packet size (the coordinate is rewritten per row read).
                noc_async_read_one_packet_set_state<true>(
                    get_noc_addr_from_bank_id<false>(0, source_base), segment_bytes, row_read_vc);
            }
            // This tile row's slot must have been popped by compute (only when the rows CB is shallower than the
            // shard).
            while (pushed_rows + rows_depth_tile_rows <= row_idx) {
                retire_next();
            }
            cb_reserve_back(cb_rows, (row_idx - pushed_rows + 1) * shard_col_tiles);
            // Slot by ring position, not by the CB write pointer (which lags while reads are in flight).
            const uint32_t slot_l1 = rows_base_l1 + (row_idx % rows_depth_tile_rows) * tile_row_bytes;
            volatile tt_l1_ptr uint32_t* ids = row_ids + row_idx * TILE_DIM;

            for (uint32_t first = 0; first < TILE_DIM; first += batch_reads) {
                while (retired + batch_depth <= issued) {
                    retire_next();
                }
                const uint32_t slot = issued % batch_depth;
                noc_async_read_set_trid(slot + 1);
#ifndef ABLATE_ROWS
                for (uint32_t i = first; i < first + batch_reads; ++i) {
                    issue_row(ids[i], slot_l1 + i * segment_bytes);
                }
#endif
                completes[slot] = (first + batch_reads == TILE_DIM) ? row_idx + 1 : 0;
                ++issued;
                weights_safe = weights_issued;  // every weight read issued so far has this batch behind it
                if constexpr (side_after_batches != 0) {
                    if (issued == side_after_batches) {
                        issue_side_reads();
                        side_issued_at = issued;
                        // Restore the row reads' one-packet size and VC (the coordinate is rewritten per read).
                        noc_async_read_one_packet_set_state<true>(
                            get_noc_addr_from_bank_id<false>(0, source_base), segment_bytes, row_read_vc);
                    }
                }

                // Weights / bias: pushed as soon as they have landed, polled without blocking the row issue. Polled
                // only here, with this batch's row reads behind them, so the id check is not racy (MOE_GLUE §4).
                poll_side_reads(false, false);
            }
        }
    }
    if (!side_started) {
        issue_side_reads();  // fewer row batches than side_after_batches: the full barrier below covers them
    }
    // The most recent reads: the full barrier.
    {
        NMC_ZONE("NMC-DRAIN");
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            poll_side_reads(false, true);  // keep delivering to the peers while the last reads land
        }
        noc_async_read_barrier();
    }
    poll_side_reads(true, false);
    poll_side_reads(true, false);  // reader 0: the bias placement, if it was issued by the first call
    while (retired < issued) {
        mark_retired();
    }
    push_landed();
    if (forwards) {
        signal_peers(true);
        noc_async_atomic_barrier();
    }
}
