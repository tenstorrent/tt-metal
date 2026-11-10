// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The row-major y writer (flat_options::y_row_major). y is [rows, H] bf16 row major, one page (H * 2 bytes) per flat
// row. The down compute pack-untilizes each row tile of its [row tiles x columns] block and pushes it as `columns`
// pages (32 rows x columns * 32 bf16, row after row); the writer sends each token row's columns * 64 bytes to
// y[row, first column * 32 ..], skipping rows at or past the entry's token count. Up to y_row_major_rows row tiles (the
// out CB's capacity) are in flight, each on its own write transaction id; they retire in order once their id is
// flushed. A sub-block is complete after its row tiles that hold tokens (the count the compute uses). The caller resets
// the write transaction id to 0 before the kernel exits.
//
// Running with the combine (flat_options::hands_y_to_combine), y is not written to DRAM: each token row goes straight
// into a landing slot of its destination combine core (CombineHandoff below).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "expert_schedule.hpp"
#include "flat_accessors.hpp"
#include "flat_options.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/dataflow/combine_formats.hpp"

// The y writer's L1 hand-off to the combine (routed_experts_combine with the flat expert's program). Each token row's
// segment goes straight into a slot of the row's destination core (a combine port's injection ring, or an
// accumulator's local ring), from a per-chip table (per 32-row tile one 256-byte page: (destination, header word) per
// row). A destination numbers its rows in the order the writers produce them (all writers walk the same schedule, so
// they agree), grants slots (cumulative grants in this core's words, less the grants of earlier calls), and takes a row
// once every writer's landed counter there covers it (+1 per row, sent when a row tile's writes are acknowledged).
// The column-0 writer also writes each row's trailer (combine_formats.hpp).
//
// A row tile goes out in one or more units: as many leading rows as the destinations have granted slots for, the rest
// once they grant more, so a ring smaller than one tile's rows for it (e.g. an accumulator's local ring of a few slots)
// never blocks the writer for good. Each unit has its own transaction id and landed counts; the tile's out CB pages go
// when its last unit retires.
namespace combine_handoff {

// The hand-off args in this core's per-core block, from flat_options::combine_handoff_arg (host: kFlatYL1Args, filled
// by the combine: combine_program.hpp):
constexpr uint32_t kArgDestinations = 0;      // destinations (0: nothing to hand off, the flat expert without y)
constexpr uint32_t kArgSlotStride = 1;        // bytes between a destination's slots
constexpr uint32_t kArgTable = 2;             // the row table's address (DRAM, flat_acc::y_table)
constexpr uint32_t kArgWords = 3;             // this core's words (layout below)
constexpr uint32_t kArgZeroGateXY = 5;        // the combine's zero gate core: noc x | noc y << 16
constexpr uint32_t kArgZeroGateWords = 6;     // its words' address (combine_zero_gate.hpp)
constexpr uint32_t kArgZeroBytes = 7;         // bytes of this core's words to zero each call (0: none)
constexpr uint32_t kArgFirstDestination = 8;  // then per destination:
constexpr uint32_t kDestinationArgs = 5;      // noc x, noc y, first slot address, slots, landed counter array address

// This core's words: grants received (cumulative), grants of earlier calls, the row table page, the trailers in flight.
constexpr uint32_t kMaxDestinations = 32;
constexpr uint32_t kWordsGrantsBefore = kMaxDestinations;  // (uint32 index)
constexpr uint32_t kWordsTablePageBytes = 256;             // (byte offset of the cached table page)
constexpr uint32_t kTablePageBytes = 256;                  // a 32-row tile's table page: 2 words a row
constexpr uint32_t kWordsTrailerStageBytes = 512;  // (byte offset of the staged trailers: a unit's 32 rows each)

// A table row: word 0 the destination (bits 0-7) and the row's gate weight (bf16 bits, bits 16-31); word 1 the
// row's header word (combine_formats.hpp)
constexpr uint32_t table_destination(uint32_t word) { return word & 0xFFu; }
constexpr uint32_t table_weight_bits(uint32_t word) { return word >> 16; }

template <bool enabled, uint32_t in_flight>
struct State {};  // (not running with the combine: nothing)

template <uint32_t in_flight>
struct State<true, in_flight> {
    uint32_t destinations = 0, slot_stride = 0, words = 0, writer_index = 0, table_address = 0;
    uint32_t cached_table_page = 0xFFFFFFFF;
    uint32_t dest_x[kMaxDestinations], dest_y[kMaxDestinations], dest_first_slot[kMaxDestinations];
    uint32_t dest_slots[kMaxDestinations], dest_landed[kMaxDestinations];
    uint32_t rows_issued[kMaxDestinations];          // rows issued per destination this call
    uint8_t unit_rows[in_flight][kMaxDestinations];  // rows per destination of each unit in flight
    uint8_t unit_ends_tile[in_flight];               // 1: the unit completes its row tile
    uint32_t tile_rows_issued = 0;                   // rows of the current row tile already issued
    uint32_t units_issued = 0, units_retired = 0;
    volatile tt_l1_ptr uint32_t* grants;         // grants received (cumulative)
    volatile tt_l1_ptr uint32_t* grants_before;  // grants consumed by earlier calls

    void init() {
        constexpr uint32_t args = flat_options::combine_handoff_arg;
        destinations = moe_block_args::arg(args + kArgDestinations);
        slot_stride = moe_block_args::arg(args + kArgSlotStride);
        table_address = moe_block_args::arg(args + kArgTable);
        words = moe_block_args::arg(args + kArgWords);
        writer_index = moe_block_args::arg(flat_options::combine_writer_index_arg);
        for (uint32_t d = 0; d < destinations; ++d) {
            const uint32_t at = args + kArgFirstDestination + kDestinationArgs * d;
            dest_x[d] = moe_block_args::arg(at);
            dest_y[d] = moe_block_args::arg(at + 1);
            dest_first_slot[d] = moe_block_args::arg(at + 2);
            dest_slots[d] = moe_block_args::arg(at + 3);
            dest_landed[d] = moe_block_args::arg(at + 4);
            rows_issued[d] = 0;
        }
        grants = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
        grants_before = grants + kWordsGrantsBefore;
        // The words are in the call's arena: zero them, then count in on the combine's zero gate, before any grant can
        // arrive (the gate releases the ports / accumulators that grant only after every writer)
        if (const uint32_t zero_bytes = moe_block_args::arg(args + kArgZeroBytes); zero_bytes != 0) {
            const uint32_t gate_xy = moe_block_args::arg(args + kArgZeroGateXY);
            const uint32_t gate_words = moe_block_args::arg(args + kArgZeroGateWords);
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t done = 0; done < zero_bytes; done += MEM_ZEROS_SIZE) {
                const uint32_t bytes = zero_bytes - done < MEM_ZEROS_SIZE ? zero_bytes - done : MEM_ZEROS_SIZE;
                noc_async_read(zeros, words + done, bytes);
            }
            noc_async_read_barrier();
            // (the gate's word 2 counts the writers in: combine_zero_gate.hpp)
            noc_semaphore_inc(get_noc_addr(gate_xy & 0xFFFF, gate_xy >> 16, gate_words + 8), 1);
            noc_async_atomic_barrier();
        }
    }
    // after the last row: the grants this call consumed
    void finish() {
        for (uint32_t d = 0; d < destinations; ++d) {
            grants_before[d] = grants_before[d] + rows_issued[d];
        }
    }
};

}  // namespace combine_handoff

// out_cb: the row-major out CB; columns: y tile columns this core writes; hidden_tiles: y tile columns in all (H / 32);
// in_flight: row tiles in flight, the out CB's capacity in row tiles; hands_off: running with the combine (y rows into
// its L1, not DRAM)
template <
    uint32_t out_cb,
    uint32_t columns,
    uint32_t hidden_tiles,
    uint32_t in_flight = flat_options::y_row_major_rows,
    bool hands_off = flat_options::hands_y_to_combine>
struct RowMajorYWriter {
    static constexpr uint32_t segment_bytes = columns * 64;     // one token row of this core's columns
    static constexpr uint32_t row_tile_bytes = columns * 2048;  // one row tile in the out CB
    static constexpr uint32_t first_trid = 8;
    // The y writes' unicast VC: not the default write VC (1), which the weight forwarding, the h chain and the read
    // requests share; a y write queued behind them there stalls this core's NIU, and through its L1 reads the down
    // matmul's unpacker.
    static constexpr uint8_t y_vc = 2;
    static constexpr uint8_t y_noc = noc_index;  // the y writes' NoC: this RISC's
    static_assert(in_flight >= 1 && first_trid + in_flight <= 16);

    decltype(TensorAccessor(flat_acc::y, 0u, 0u)) y;
    uint32_t column_offset;
    uint32_t base = 0;  // the out CB's start (its read pointer before anything is consumed)
    // the issue cursor (entry, sub-block, row tile) and the in-order retire cursor
    uint32_t issue_entry = 0, issue_sub_block = 0, issue_row_tile = 0, issued = 0;
    uint32_t retire_entry = 0, retire_sub_block = 0, retire_row_tile = 0, retired = 0;
    uint32_t done_entry = 0;  // the entry of the sub-block the last retire() completed
    combine_handoff::State<hands_off, in_flight> handoff;

    RowMajorYWriter(uint32_t y_address, uint32_t first_column) :
        y(TensorAccessor(flat_acc::y, y_address, hidden_tiles * 64)),
        column_offset(first_column * 64),
        base(get_read_ptr(out_cb)) {
        if constexpr (hands_off) {
            handoff.init();
        }
    }
    // after the last row
    void finish() {
        if constexpr (hands_off) {
            handoff.finish();
        }
    }

    static uint32_t row_tiles(const ExpertSchedule& schedule, uint32_t entry, uint32_t sub_block) {
        const uint32_t left = schedule.tokens[entry] - sub_block * flat_options::rows_per_sub_block;
        return left >= flat_options::rows_per_sub_block ? kSubBlockRowTiles : (left + 31) / 32;
    }

    // Issue the next row tile's writes if the compute has pushed it and a transaction id is free; true if issued.
    bool issue(const ExpertSchedule& schedule) {
        auto tile_ready = [&]() {
            return issued - retired < in_flight &&
                   cb_pages_available_at_front(out_cb, (issued - retired + 1) * columns);
        };
        if constexpr (hands_off) {  // (a tile partly handed off is ready)
            if (handoff.units_issued - handoff.units_retired == in_flight ||
                (handoff.tile_rows_issued == 0 && !tile_ready())) {
                return false;
            }
        } else if (!tile_ready()) {
            return false;
        }
        const uint32_t source = base + (issued % in_flight) * row_tile_bytes;
        const uint32_t entry_row = (issue_sub_block * kSubBlockRowTiles + issue_row_tile) * 32;
        const uint32_t tokens_left = schedule.tokens[issue_entry] - entry_row;
        const uint32_t rows = tokens_left < 32 ? tokens_left : 32;
        const uint32_t first_row = schedule.first_row[issue_entry] + entry_row;
        if constexpr (hands_off) {
            const UnitIssued unit = issue_handoff_unit(source, first_row, rows);
            if (unit == UnitIssued::None) {
                return false;
            }
            if (unit == UnitIssued::PartOfTile) {  // the tile's other rows in a later unit: the cursor stays on it
                return true;
            }
        } else {
            const uint32_t trid = first_trid + issued % in_flight;
            for (uint32_t row = 0; row < rows; ++row) {
                noc_async_write_one_packet_with_trid(
                    source + row * segment_bytes,
                    y.get_noc_addr(first_row + row, column_offset, y_noc),
                    segment_bytes,
                    trid,
                    write_cmd_buf,
                    y_noc,
                    y_vc);
            }
        }
        ++issued;
        if (++issue_row_tile == row_tiles(schedule, issue_entry, issue_sub_block)) {
            issue_row_tile = 0;
            if (++issue_sub_block == schedule.sub_blocks[issue_entry]) {
                issue_sub_block = 0;
                ++issue_entry;
            }
        }
        return true;
    }

    // Retire the oldest issued row tile once its writes are flushed (frees its pages); returns the number of
    // sub-blocks completed by it (0 or 1).
    uint32_t retire(const ExpertSchedule& schedule) {
        if constexpr (hands_off) {
            const uint32_t unit = handoff.units_retired % in_flight;
            if (handoff.units_retired == handoff.units_issued ||
                !ncrisc_noc_nonposted_write_with_transaction_id_flushed(y_noc, first_trid + unit)) {
                return 0;
            }
            for (uint32_t d = 0; d < handoff.destinations; ++d) {
                if (handoff.unit_rows[unit][d]) {
                    const uint64_t landed = get_noc_addr(
                        handoff.dest_x[d], handoff.dest_y[d], handoff.dest_landed[d] + 4 * handoff.writer_index, y_noc);
                    noc_semaphore_inc(landed, handoff.unit_rows[unit][d], y_noc);
                }
            }
            const bool tile_done = handoff.unit_ends_tile[unit] != 0;
            ++handoff.units_retired;
            if (!tile_done) {
                return 0;
            }
        } else {
            if (retired == issued ||
                !ncrisc_noc_nonposted_write_with_transaction_id_flushed(y_noc, first_trid + retired % in_flight)) {
                return 0;
            }
        }
        cb_pop_front(out_cb, columns);
        ++retired;
        if (++retire_row_tile == row_tiles(schedule, retire_entry, retire_sub_block)) {
            retire_row_tile = 0;
            done_entry = retire_entry;
            if (++retire_sub_block == schedule.sub_blocks[retire_entry]) {
                retire_sub_block = 0;
                ++retire_entry;
            }
            return 1;
        }
        return 0;
    }

private:
    enum class UnitIssued { None, PartOfTile, WholeTile };

    // The L1 hand-off of the current row tile's next unit: its next rows while their destinations have granted slots.
    UnitIssued issue_handoff_unit(uint32_t source, uint32_t first_row, uint32_t rows) {
        namespace fmt = combine_formats;
        using namespace combine_handoff;
        auto& state = handoff;
        const uint32_t unit = state.units_issued % in_flight;
        const uint32_t trid = first_trid + unit;
        uint8_t* unit_rows = state.unit_rows[unit];
        if (state.destinations == 0) {  // nothing to hand off to: the flat expert alone without y
            for (uint32_t d = 0; d < kMaxDestinations; ++d) {
                unit_rows[d] = 0;
            }
            state.unit_ends_tile[unit] = 1;
            ++state.units_issued;
            return UnitIssued::WholeTile;
        }
        // this tile's table page: (destination, header word) per row
        const uint32_t page = first_row / 32, table_l1 = state.words + kWordsTablePageBytes;
        if (page != state.cached_table_page) {
            const auto table = TensorAccessor(flat_acc::y_table, state.table_address, kTablePageBytes);
            noc_async_read(table.get_noc_addr(page), table_l1, kTablePageBytes);
            noc_async_read_barrier();
            state.cached_table_page = page;
        }
        const volatile tt_l1_ptr uint32_t* table_rows = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(table_l1);
        for (uint32_t d = 0; d < state.destinations; ++d) {
            unit_rows[d] = 0;
        }
        invalidate_l1_cache();
        // the unit: the tile's next rows while their destinations have granted slots for them
        uint32_t unit_end = state.tile_rows_issued;
        for (; unit_end < rows; ++unit_end) {
            const uint32_t d = table_destination(table_rows[2 * unit_end]);
            if (state.grants[d] - state.grants_before[d] < state.rows_issued[d] + unit_rows[d] + 1) {
                break;
            }
            ++unit_rows[d];
        }
        if (unit_end == state.tile_rows_issued) {
            return UnitIssued::None;
        }
        // the trailers in flight (column-0 writer): 32 rows of a unit each
        const uint32_t stage = state.words + kWordsTrailerStageBytes + unit * 32 * fmt::kTrailerBytes;
        if (column_offset == 0) {
            for (uint32_t row = state.tile_rows_issued; row < unit_end; ++row) {
                volatile tt_l1_ptr uint32_t* trailer =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + row * fmt::kTrailerBytes);
                for (uint32_t word = 0; word < fmt::kTrailerBytes / 4; ++word) {
                    trailer[word] = 0;
                }
                trailer[fmt::kTrailerHeader] = table_rows[2 * row + 1];
                trailer[fmt::kTrailerWeight] = table_weight_bits(table_rows[2 * row]);
                trailer[fmt::kTrailerFlagLow] = 1;
                trailer[fmt::kTrailerFlagHigh] = 1;
            }
            // (the NoC reads the stage straight from L1: these stores must land first, or a trailer write can carry
            // the stage's previous words - e.g. a zeroed flag (the stage is zeroed every call), and the row is never
            // taken at its home: the combine hangs)
            asm volatile("fence" ::: "memory");
        }
        for (uint32_t row = state.tile_rows_issued; row < unit_end; ++row) {
            const uint32_t d = table_destination(table_rows[2 * row]);
            const uint32_t slot =
                state.dest_first_slot[d] + (state.rows_issued[d] % state.dest_slots[d]) * state.slot_stride;
            ++state.rows_issued[d];
            noc_async_write_one_packet_with_trid(
                source + row * segment_bytes,
                get_noc_addr(state.dest_x[d], state.dest_y[d], slot + column_offset, y_noc),
                segment_bytes,
                trid,
                write_cmd_buf,
                y_noc,
                y_vc);
            if (column_offset == 0) {  // its trailer, staged above, after the row's H bf16
                noc_async_write_one_packet_with_trid(
                    stage + row * fmt::kTrailerBytes,
                    get_noc_addr(state.dest_x[d], state.dest_y[d], slot + hidden_tiles * 64, y_noc),
                    fmt::kTrailerBytes,
                    trid,
                    write_cmd_buf,
                    y_noc,
                    y_vc);
            }
        }
        state.unit_ends_tile[unit] = unit_end == rows ? 1 : 0;
        ++state.units_issued;
        if (unit_end < rows) {
            state.tile_rows_issued = unit_end;
            return UnitIssued::PartOfTile;
        }
        state.tile_rows_issued = 0;
        return UnitIssued::WholeTile;
    }
};

// The y writer of a down weight reader that does not write y (the BRISC writes bfp8 tiles): does nothing.
struct NoYWriter {
    uint32_t done_entry = 0;
    NoYWriter(uint32_t, uint32_t) {}
    void finish() {}
    bool issue(const ExpertSchedule&) { return false; }
    uint32_t retire(const ExpertSchedule&) { return 0; }
};
