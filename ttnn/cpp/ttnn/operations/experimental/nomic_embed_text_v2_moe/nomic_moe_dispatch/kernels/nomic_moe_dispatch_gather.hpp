// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_dispatch: the gather reader body, shared by both reader RISCs (op_design.md, regime R1).
//   nomic_moe_dispatch_reader.cpp      RISCV_0 / NOC0: rows [row_begin, row_end) = [0, rows_noc0) of every block
//   nomic_moe_dispatch_reader_noc1.cpp RISCV_1 / NOC1: rows [rows_noc0, 32) of every block
//
// Per core c (c = y * grid_x + x, the ND spec's round-robin order) the blocks are shards s = c + num_cores * j,
// j = 0 .. num_blocks - 1; shard s is tile row s / k, column piece s % k (piece_bytes of each row).
//
// Ids (Refinement 3, R10 `ids_row_mcast`): a block's ids are read from the core's ids scratch, a full-size image of
//   gather_index at the same address on every core (tile row t's 32 ids at t * 128 B). One grid row's 11 cores hold 11
//   consecutive shards in every round j (block j of each), so the row's round-j ids are one contiguous span of tile
//   rows. The row's sender (the ids_sender_kernel reader -- NOC0 by default -- of the row's x = 0 core, which holds
//   the row's most blocks) reads that
//   span from the one gather_index DRAM bank and multicasts it along its row (a path no other row's multicast uses),
//   then its round count: it writes count r + 1 to a cell of its own (never rewritten) and multicasts it to the row's
//   ids_ready semaphore, after the data on the same NoC and VC, and sets its own ids_ready. Every reader waits
//   ids_ready > j before block j. Rounds 0 and 1 are fetched and published first thing; round j + 2 is published at
//   the sender's block j start, fetched ids_lookahead blocks before that, so both land under the row reads.
//   Measured against the alternatives (changelog, Refinement 3): one 64 B DRAM read a reader a block put 220 requests
//   on the one bank per round (the first block's ids landed 0.5-2.0 us in); multicasting the index to the whole grid
//   costs ~0.5 us a 7 KB chunk idle and ~1-1.25 us behind the gather's traffic, and grid multicasts serialize.
// gather_rows_block: one stateful one-packet read a row (only coordinate, source and destination are written per
//   read), tagged with the block's trid (a ring of ROW_TRIDS), into the block's slot of the rows CB. At most
//   reads_in_flight reads outstanding (the host caps it by bytes too). A block is pushed (w pages) once its trid is
//   flushed. Issue stagger (Refinement 3): each core walks its half's rows from its own start row in its own odd step;
//   with ids local the whole grid starts at once, and cores walking sorted ids in lockstep meet on the same banks.
//
// Issue path and row split (Refinement 4): the readers are RISC-issue-bound -- with every row read stubbed (address
// math
//   kept) a block body still took ~0.85 us of the ~1.1 -- so the per-row path folds every NIU register address at
//   compile time (noc_id = noc_index), reads the response counter only when the credits from its last reading are
//   spent, and reads the ids non-volatile after the ready wait. The rows of a block are split positionally ([row_begin,
//   row_end)); on grid column 0 the host gives the ids sender reader fewer rows (its own reader kernels, sender shed).
//   The path-aware split (row_split_path: each row to the NoC with the shorter response path, both readers walking all
//   32 ids) is a parked knob: its per-block owner pass sits on the issue path and cost more than the assignment saved.
//
// Helper substitution (documented per the kernel-lib policy): the per-read path is raw NoC API because
// dataflow_kernel_lib::read_with_state (local_copy_helpers_dataflow.hpp:57, :74-95) and Noc::async_read_with_state
// (api/dataflow/noc.h:295-379) fix the source NoC coordinate at set-state time, and every read here has its own
// source bank; noc.async_read(TensorAccessor, ...) pays generic address generation per read (60-70 cycles, the
// incumbent's issue bound). The ids DRAM reads use noc.async_read with the TensorAccessor.
// The ids row multicast is raw multicast + semaphore too: the mcast helper's Counter receivers are counted by an atomic
// the source core never gets (so the sender core's NOC1 reader, which also waits for the ids, would starve), the host
// attachment refuses a second kernel on the sender core with a different NoC, and its Flag receivers clear the flag
// (one consumer a core; here both reader RISCs poll the one monotone count).
//
// Transaction-id race (MOE_GLUE.md section 4): a trid's outstanding count can read 0 before the most recent read under
// it has left the command buffer. A block's trid is polled only once a read of a later block has been issued (that
// issue waited for the command buffer); the most recent block is covered by the full read barrier. An ids read is
// waited for by trid only after the command buffer has drained (wait_trid).

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"

namespace nomic_moe_dispatch {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t ID_BYTES = sizeof(uint32_t);
constexpr uint32_t ROW_TRID_BASE = 3;
constexpr uint32_t ROW_TRIDS = 4;  // row-block trids 3 .. 6
constexpr uint32_t IDS_TRID_BASE = 7;
constexpr uint32_t IDS_TRIDS = 8;  // ids read trids 7 .. 14 (NOC_MAX_TRANSACTION_ID 0xF)

FORCE_INLINE uint32_t row_trid(uint32_t block) { return ROW_TRID_BASE + block % ROW_TRIDS; }

FORCE_INLINE bool trid_flushed(uint32_t noc, uint32_t trid) {
    return ncrisc_noc_read_with_transaction_id_flushed(noc, trid);
}

FORCE_INLINE uint32_t reads_outstanding(uint32_t noc) {
    return noc_reads_num_issued[noc] - NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
}

FORCE_INLINE void wait_trid(uint32_t noc, uint32_t trid) {
    // The trid race: poll a trid only once the read command buffer has drained (every issued read has left it).
    while (!noc_cmd_buf_ready(noc, read_cmd_buf)) {
    }
    while (!trid_flushed(noc, trid)) {
    }
}

inline void gather_reader() {
    constexpr uint32_t w = get_named_compile_time_arg_val("block_width_tiles");
    constexpr uint32_t num_shards = get_named_compile_time_arg_val("num_shards");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
    constexpr uint32_t cb_rows_id = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t cb_gather_ids_id = get_named_compile_time_arg_val("cb_gather_ids");
    constexpr uint32_t row_begin = get_named_compile_time_arg_val("row_begin");
    constexpr uint32_t row_end = get_named_compile_time_arg_val("row_end");
    constexpr uint32_t piece_bytes = get_named_compile_time_arg_val("piece_bytes");
    constexpr uint32_t table_page_bytes = get_named_compile_time_arg_val("table_page_bytes");
    constexpr uint32_t pieces_per_row = get_named_compile_time_arg_val("pieces_per_row");
    constexpr uint32_t row_slots = get_named_compile_time_arg_val("row_slots");
    constexpr uint32_t reads_in_flight = get_named_compile_time_arg_val("reads_in_flight");
    constexpr uint32_t ids_align = get_named_compile_time_arg_val("ids_align");
    constexpr uint32_t ids_total_bytes = get_named_compile_time_arg_val("ids_total_bytes");
    constexpr uint32_t ids_lookahead = get_named_compile_time_arg_val("ids_lookahead");
    constexpr uint32_t ids_ready_sem = get_named_compile_time_arg_val("ids_ready_sem");
    constexpr bool ids_sender_kernel = get_named_compile_time_arg_val("ids_sender_kernel") != 0;
    constexpr uint32_t mcast_start_x = get_named_compile_time_arg_val("mcast_start_x");
    constexpr uint32_t mcast_end_x = get_named_compile_time_arg_val("mcast_end_x");
    constexpr uint32_t row_stagger = get_named_compile_time_arg_val("row_stagger");
    constexpr uint32_t row_strides = get_named_compile_time_arg_val("row_strides");
    constexpr uint32_t start_spread_cycles = get_named_compile_time_arg_val("start_spread_cycles");
    constexpr bool row_split_path = get_named_compile_time_arg_val("row_split_path") != 0;
    constexpr uint32_t row_split_slack = get_named_compile_time_arg_val("row_split_slack");
    constexpr uint32_t reader_half = get_named_compile_time_arg_val("reader_half");
    constexpr uint32_t rows_noc0 = get_named_compile_time_arg_val("rows_noc0");
    constexpr auto index_args = TensorAccessorArgs<0>();

    static_assert(piece_bytes == w * TILE_ROWS * 2, "a row piece is w tiles of 32 bf16 values");
    static_assert(piece_bytes <= NOC_MAX_BURST_SIZE, "one-packet reads");
    static_assert(row_begin < row_end && row_end <= TILE_ROWS, "a reader owns a non-empty row range of a block");
    static_assert(row_slots >= 1 && reads_in_flight >= 1);
    static_assert(ids_total_bytes % ids_align == 0 && (TILE_ROWS * ID_BYTES) % ids_align == 0, "aligned id reads");
    static_assert(ids_lookahead >= 1 && 2 + ids_lookahead <= IDS_TRIDS, "at most IDS_TRIDS ids rounds in flight");
    static_assert(num_cores % grid_x == 0 && grid_x >= 2);
    static_assert(reader_half <= 1 && 0 < rows_noc0 && rows_noc0 < TILE_ROWS);
    static_assert(rows_noc0 + row_split_slack <= TILE_ROWS && TILE_ROWS - rows_noc0 + row_split_slack <= TILE_ROWS);

    constexpr uint32_t slot_bytes = TILE_ROWS * piece_bytes;  // one block: 32 rows x piece_bytes = w tiles
    constexpr uint32_t tile_row_id_bytes = TILE_ROWS * ID_BYTES;
    constexpr uint32_t half_rows = row_end - row_begin;
    constexpr uint32_t drain_depth = row_slots < ROW_TRIDS ? row_slots : ROW_TRIDS;
    constexpr uint32_t cmd_buf = read_cmd_buf;

    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    if (core >= num_shards) {
        return;  // no shard on this core (and none on the rest of its row: a row sender always has one)
    }
    const uint32_t num_blocks = (num_shards - core + num_cores - 1) / num_cores;
    const uint32_t table_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t index_addr = get_common_arg_val<uint32_t>(1);

    const Noc noc;
    // The reader's NoC is a compile-time constant: every NIU register address of the per-row issue folds to a constant.
    constexpr uint32_t noc_id = noc_index;
    DataflowBuffer rows_cb(cb_rows_id);
    DataflowBuffer ids_cb(cb_gather_ids_id);
    // The ids scratch is at the same address on every core (one CB layout for the grid): the multicast target. The
    // row sender's round-count cells (one ids_align cell a round) follow the gather_index image.
    const uint32_t ids_base = (ids_cb.get_write_ptr() + ids_align - 1) & ~(ids_align - 1);
    volatile tt_l1_ptr uint32_t* ids_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ids_ready_sem));

    // ---- the row sender: round r's ids (tile rows of shards [row_first + num_cores * r, + grid_x)) ----
    const uint32_t row_first = core - core % grid_x;
    const bool is_sender = ids_sender_kernel && core == row_first;
    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto round_span = [&](uint32_t r, uint32_t& offset, uint32_t& bytes) {
        const uint32_t first = row_first + num_cores * r;
        const uint32_t last = first + grid_x - 1 < num_shards ? first + grid_x - 1 : num_shards - 1;
        offset = (first / pieces_per_row) * tile_row_id_bytes;
        bytes = (last / pieces_per_row + 1) * tile_row_id_bytes - offset;
    };
    const auto ids_trid = [](uint32_t r) { return IDS_TRID_BASE + r % IDS_TRIDS; };
    const auto fetch_round = [&](uint32_t r) {
        uint32_t offset, bytes;
        round_span(r, offset, bytes);
        noc.async_read<NocOptions::TXN_ID>(
            index_accessor,
            CoreLocalMem<uint32_t>(ids_base + offset),
            bytes,
            {.page_id = 0, .offset_bytes = offset},
            {},
            {.trid = ids_trid(r)});
    };
    const uint32_t row_y = my_y[noc_id];  // the sender's own row, in its NoC's coordinates
    const uint32_t sem = reinterpret_cast<uint32_t>(ids_ready);
    const auto publish_round = [&](uint32_t r) {
        uint32_t offset, bytes;
        round_span(r, offset, bytes);
        const uint32_t addr = ids_base + offset;
        noc_async_write_multicast(
            addr,
            get_noc_multicast_addr(mcast_start_x, row_y, mcast_end_x, row_y, addr, noc_id),
            bytes,
            grid_x - 1,
            /*linked=*/true,
            noc_id);
        // The count goes out from its own cell, never rewritten: no publish waits for an earlier one to leave L1.
        const uint32_t cell = ids_base + ids_total_bytes + r * ids_align + sem % ids_align;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cell) = r + 1;
        noc_semaphore_set_multicast(
            cell,
            get_noc_multicast_addr(mcast_start_x, row_y, mcast_end_x, row_y, sem, noc_id),
            grid_x - 1,
            /*linked=*/false,
            noc_id);
        *ids_ready = r + 1;  // this core's other reader: the round is in local L1 already
    };
    // Rounds published before block j: [0, j + 2) (one block ahead); fetched: [0, j + 2 + ids_lookahead).
    if (is_sender) {
        for (uint32_t r = 0; r < num_blocks && r < 2 + ids_lookahead; ++r) {
            fetch_round(r);
        }
        noc.async_read_barrier();
        for (uint32_t r = 0; r < num_blocks && r < 2; ++r) {
            publish_round(r);
        }
    }

    // gather_rows_block state: one-packet reads of piece_bytes; per read only coordinate / src / dst are written.
    // An ids read rewrites the shared read command buffer's length and target, so the state is re-programmed after
    // each one.
    const auto program_row_state = [&]() {
        ncrisc_noc_read_set_state<DM_DEDICATED_NOC, true>(
            noc_id, cmd_buf, get_noc_addr_from_bank_id<false>(0, table_addr, noc_id), piece_bytes);
    };
    program_row_state();

    // Issue stagger: this core walks its half's rows from row_start in steps of row_step (coprime with the half, so
    // every row once), both per core: lockstep cores reading their sorted ids in the same order would otherwise meet on
    // the same banks at every step.
    static_assert(row_strides >= 1 && 2 * row_strides - 1 <= half_rows);  // the host checks each step is coprime
    // Path mode walks all 32 rows of a block (any odd step is coprime with 32), the two readers half a block apart.
    const uint32_t row_start = row_split_path ? (core * row_stagger + reader_half * (TILE_ROWS / 2)) % TILE_ROWS
                                              : (core * row_stagger) % half_rows;
    const uint32_t row_step = 2 * (core % row_strides) + 1;

    if constexpr (start_spread_cycles > 0) {
        // Start stagger: core c's first block waits a spread fraction set by a scrambled core rank.
        riscv_wait((core * 37 % num_cores) * start_spread_cycles / num_cores);
    }
    // Path-aware row split (Refinement 4, lamp L1): a row goes to the reader whose NoC carries its response (bank ->
    // this core, the data-carrying packet) over fewer torus hops. NOC0 routes +x / +y, NOC1 -x / -y, so on each axis
    // the two response paths are d and size - d (d = (me - bank) mod size): the score (NOC0 hops - NOC1 hops) separates
    // into a per-column and a per-row term, built once here (29 entries) in the NOC0 frame on both readers (my_x[0] /
    // my_y[0] and l1_bank_to_noc_xy[0], so both readers derive the same split). score > 0: NOC1 is shorter.
    int8_t x_score[row_split_path ? noc_size_x : 1];
    int8_t y_score[row_split_path ? noc_size_y : 1];
    if constexpr (row_split_path) {
        const auto axis_score = [](uint32_t me, uint32_t other, uint32_t size) -> int8_t {
            const uint32_t d0 = me >= other ? me - other : me + size - other;  // NOC0 response hops on this axis
            return d0 == 0 ? 0 : static_cast<int8_t>(2 * static_cast<int32_t>(d0) - static_cast<int32_t>(size));
        };
        for (uint32_t x = 0; x < noc_size_x; ++x) {
            x_score[x] = axis_score(my_x[0], x, noc_size_x);
        }
        for (uint32_t y = 0; y < noc_size_y; ++y) {
            y_score[y] = axis_score(my_y[0], y, noc_size_y);
        }
    }
    // A block's split: rows in order 0 .. 31 go to their shorter-path reader (ties alternate) until that reader holds
    // its share + row_split_slack rows, then to the other. Both readers compute the same split from the same ids.
    constexpr uint32_t cap0 = rows_noc0 + row_split_slack;
    constexpr uint32_t cap1 = TILE_ROWS - rows_noc0 + row_split_slack;
    uint32_t row_src[row_split_path ? TILE_ROWS : 1];
    uint32_t row_coord[row_split_path ? TILE_ROWS : 1];

    const uint32_t slot_base = rows_cb.get_write_ptr();  // the rows CB's base: nothing pushed yet
    uint32_t pushed = 0;                                 // blocks pushed to the rows CB
    uint32_t pollable = 0;  // blocks with a read of a later block issued behind them (safe to poll by trid)
    uint32_t issued = 0;    // blocks whose reads are all issued

    const auto try_push = [&]() {
        while (pushed < pollable && trid_flushed(noc_id, row_trid(pushed))) {
            rows_cb.push_back(w);
            ++pushed;
        }
    };
    // Push until `target` blocks are pushed; the most recent block needs the full barrier (the trid race).
    const auto drain_to = [&](uint32_t target) {
        while (pushed < target) {
            if (pushed < pollable) {
                try_push();
            } else {
                noc_async_read_barrier(noc_id);
                pollable = issued;
                try_push();
            }
        }
    };

    // The per-row issue (Refinement 4): the readers are RISC-issue-bound (with every row read stubbed, a block's issue
    // loop alone took ~0.85 of the ~1.1 us a block), so the per-row path reads no NIU register it can avoid. The reads
    // in flight are tracked by credits: the response counter is read only when the credits from its last reading are
    // spent (one register read per up to reads_in_flight rows instead of one per row). Then the command buffer poll,
    // the target coordinate and the stateful one-packet read.
    uint32_t credits = 0;
    const auto issue_row = [&](uint32_t coord, uint32_t src, uint32_t dst) {
        if (credits == 0) {
            uint32_t outstanding;
            while ((outstanding = reads_outstanding(noc_id)) >= reads_in_flight) {
                try_push();
            }
            credits = reads_in_flight - outstanding;
        }
        --credits;
        while (!noc_cmd_buf_ready(noc_id, cmd_buf)) {
        }
        noc_cmd_buf_set_targ_addr_coordinate(noc_id, cmd_buf, coord);
        ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, true, true>(noc_id, cmd_buf, src, dst);
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t shard = core + num_cores * block_idx;
        const uint32_t tile_row = shard / pieces_per_row;
        if (is_sender) {
            const uint32_t publish = block_idx + 2;
            if (publish < num_blocks) {
                wait_trid(noc_id, ids_trid(publish));
                publish_round(publish);
            }
            if (publish + ids_lookahead < num_blocks) {
                fetch_round(publish + ids_lookahead);
                program_row_state();
            }
        }
        // This block's ids: round block_idx of this core's row (a no-op once it has landed).
        noc_semaphore_wait_min(ids_ready, block_idx + 1);
        asm volatile("" ::: "memory");  // the block's ids are read (non-volatile, schedulable) only after the wait

        // A block's trid and slot are reused only after the block that held them was pushed (and, for the slot,
        // tilized: reserve_back waits for compute's pops).
        if (block_idx + 1 > drain_depth) {
            drain_to(block_idx + 1 - drain_depth);
        }
        rows_cb.reserve_back((block_idx - pushed + 1) * w);
        ncrisc_noc_set_transaction_id(noc_id, cmd_buf, row_trid(block_idx));

        const uint32_t piece_offset = (shard - tile_row * pieces_per_row) * piece_bytes;
        const uint32_t src_base = table_addr + piece_offset;
        const tt_l1_ptr uint32_t* ids =
            reinterpret_cast<const tt_l1_ptr uint32_t*>(ids_base + tile_row * tile_row_id_bytes);
        const uint32_t slot = slot_base + (block_idx % row_slots) * slot_bytes;

        if constexpr (row_split_path) {
            uint32_t mine = 0;  // this reader's rows of the block
            uint32_t count0 = 0, count1 = 0;
            for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                const uint32_t id = ids[r];
                const uint32_t page_in_bank = udivsi3_const_divisor<NUM_L1_BANKS>(id);
                const uint32_t bank = id - page_in_bank * NUM_L1_BANKS;
                const uint32_t xy0 = l1_bank_to_noc_xy[0][bank];
                const int32_t score = x_score[NOC_UNICAST_COORDINATE_X(xy0)] + y_score[NOC_UNICAST_COORDINATE_Y(xy0)];
                bool to1 = score > 0 || (score == 0 && count1 < count0);
                if (to1 ? count1 >= cap1 : count0 >= cap0) {
                    to1 = !to1;
                }
                to1 ? ++count1 : ++count0;
                if (to1 == (reader_half == 1)) {
                    mine |= 1u << r;
                    row_src[r] = src_base + page_in_bank * table_page_bytes + bank_to_l1_offset[bank];
                    row_coord[r] = l1_bank_to_noc_xy[noc_id][bank];
                }
            }
            bool first = true;
            uint32_t row = row_start;
            for (uint32_t i = 0; i < TILE_ROWS; ++i) {
                if ((mine >> row) & 1) {
                    issue_row(row_coord[row], row_src[row], slot + row * piece_bytes);
                    if (first) {
                        pollable = block_idx;  // the previous block's last read has left the command buffer
                        first = false;
                    }
                }
                row = (row + row_step) & (TILE_ROWS - 1);
            }
            issued = block_idx + 1;
            try_push();
            continue;
        }
        uint32_t row = row_begin + row_start;
        for (uint32_t i = 0; i < half_rows; ++i) {
            const uint32_t id = ids[row];
            const uint32_t page_in_bank = udivsi3_const_divisor<NUM_L1_BANKS>(id);
            const uint32_t bank = id - page_in_bank * NUM_L1_BANKS;
            const uint32_t src = src_base + page_in_bank * table_page_bytes + bank_to_l1_offset[bank];
            const uint32_t coord = l1_bank_to_noc_xy[noc_id][bank];
            issue_row(coord, src, slot + row * piece_bytes);
            if (i == 0) {
                pollable = block_idx;  // the previous block's last read has left the command buffer
            }
            row += row_step;
            if (row >= row_end) {
                row -= half_rows;
            }
        }
        issued = block_idx + 1;
        try_push();
    }

    // The most recent reads: the full barrier, then the remaining pushes.
    noc_async_read_barrier(noc_id);
    pollable = issued;
    while (pushed < num_blocks) {
        rows_cb.push_back(w);
        ++pushed;
    }
    ncrisc_noc_set_transaction_id(noc_id, cmd_buf, 0);  // leave the read command buffer untagged
    if (is_sender) {
        noc_async_write_barrier(noc_id);  // the ids multicasts' acks
    }
}

}  // namespace nomic_moe_dispatch
