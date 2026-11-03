// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// linear_gelu reader (NCRISC, NoC0).
// load_activation_block: x row-block (rows x K tiles) plus the ones tile at K position k_tiles of every row,
//   read by the round's sender of the grid row (one barrier) and multicast along the row (regime R2);
//   nominal block_h * k_ext_tiles pages pushed.
// deliver_weight_block: chunk j of the column's W slice (W rows 0..k_tiles-1 plus the bias tile row, block_w
//   columns, laid out [j][k][w] in cb_weight_resident) is owned by the column's core of rotating round
//   w_round_offset + j. The owner issues its DRAM read at kernel start for j < w_prefetch_rounds, else in round j,
//   zeroes the bias tiles' padding rows (they multiply the ones tile's zero columns) and multicasts it along the
//   column; the rest receive. Never popped.
// Order: owned W reads of rounds < w_prefetch_rounds, x row-block 0 reads issued (sender), W chunks
// < w_chunks_before_x, x row-block 0 multicast, remaining W chunks, remaining row-blocks.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_activation_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_resident = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_ones_tile = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr uint32_t block_w = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t k_ext_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_prefetch_rounds = get_compile_time_arg_val(9);
    // W chunks delivered before x row-block 0's multicast (0 = x first).
    constexpr uint32_t w_chunks_before_x = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto w_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wmc_ct_offset"), get_named_compile_time_arg_val("wmc_rt_offset")>();
    constexpr auto x_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xmc_ct_offset"), get_named_compile_time_arg_val("xmc_rt_offset")>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t core_rows = get_arg_val<uint32_t>(4);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t num_n_blocks = get_arg_val<uint32_t>(6);
    // Added to the x round so grid rows rotate senders out of phase (diagonal), spreading read responses over
    // the columns' NoC links instead of funnelling them all into one column.
    const uint32_t x_round_offset = get_arg_val<uint32_t>(7);
    // Same for the W rounds of a column (chunk owner = round % column rows); see W_ROUND_SHIFT on the host.
    const uint32_t w_round_offset = get_arg_val<uint32_t>(8);

    static_assert(k_ext_tiles == k_tiles + 1, "one bias K tile");
    // X1_TRID: x and W reads carry separate NoC transaction ids and each completion waits on its own id only, so
    // a W owner's barrier does not also wait for x row-block reads issued early (and vice versa).
#ifdef X1_TRID
    constexpr uint32_t x_trid = 1;
    constexpr uint32_t w_trid = 2;
    auto tag_reads = [](uint32_t trid) { noc_async_read_set_trid(trid); };
    auto wait_reads = [](uint32_t trid) { noc_async_read_barrier_with_trid(trid); };
#else
    constexpr uint32_t x_trid = 0;
    constexpr uint32_t w_trid = 0;
    auto tag_reads = [](uint32_t) {};
    auto wait_reads = [](uint32_t) { noc_async_read_barrier(); };
#endif
    // X1_NOC1: x reads of row-blocks >= 1 go out on NoC1 (the writer issues no reads there), so their responses do
    // not share NoC0 links with the W column multicast that they overlap. In DM_DEDICATED_NOC mode the firmware
    // only seeds NCRISC's software read counters for its own NoC, so completion is tracked with the hardware
    // per-transaction-id outstanding counter on NoC1 instead of noc_async_read_barrier(1).
#ifdef X1_NOC1
    constexpr uint8_t x_late_noc = 1;
    constexpr uint32_t x_late_trid = 3;
#else
    constexpr uint8_t x_late_noc = noc_index;
    constexpr uint32_t x_late_trid = 0;
#endif
    static_assert(block_h * k_ext_tiles < NOC_MAX_TRANSACTION_ID_COUNT, "per-trid outstanding read counter");
    constexpr uint32_t act_block_tiles = block_h * k_ext_tiles;
    constexpr uint32_t w_chunk_tiles = k_ext_tiles * block_w;
    constexpr uint32_t face_bytes = tile_bytes / 4;
    constexpr uint32_t face_row_bytes = face_bytes / 16;

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto w_accessor = TensorAccessor(w_args, w_addr, tile_bytes);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, tile_bytes);

    Noc noc;
    CircularBuffer ones_cb(cb_bias_ones_tile);
    CircularBuffer weight_cb(cb_weight_resident);

    // Ones tile: column 0 = 1.0 (faces 0 and 2), else 0. Reader-private constant, no CB credit moves.
    const uint32_t ones_addr = get_write_ptr(cb_bias_ones_tile);
    noc.async_write_zeros(ones_cb, tile_bytes, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        constexpr uint16_t bf16_one = 0x3F80;
        volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_addr);
        constexpr uint32_t face_elems = face_bytes / sizeof(uint16_t);
        for (uint32_t r = 0; r < 16; ++r) {
            ones[r * 16] = bf16_one;
            ones[2 * face_elems + r * 16] = bf16_one;
        }
    }

    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;

    // R2: one core of the grid row reads x row-block i (round = row offset + i, rotating sender) and multicasts the
    // whole slot, ones tiles included, into the same cb_activation_block slot on the rest of the row. Every core of a
    // row has the same rows, so all walk the same slot sequence.
    auto x_sender = x_mcast.optional_sender(noc);
    auto x_receiver = x_mcast.optional_receiver(noc);
    // issue_activation_reads reserves row-block i's slot and, on the round's sender, issues its reads (no barrier);
    // finish_activation_block completes the round. Split so row-block 0's reads overlap W chunk 0's delivery.
    // Rows [row_lo, row_hi) of the row-block only (clamped to its rows); row_lo == 0 reserves the slot and the
    // piece reaching the last row issues the ones tiles.
    auto issue_activation_rows = [&](uint32_t row_block_idx, uint32_t row_lo, uint32_t row_hi) {
        MaybeDeviceZoneScope("reader_x_issue");
        const uint32_t first_row = row_block_idx * block_h;
        const uint32_t rows = (core_rows - first_row) < block_h ? (core_rows - first_row) : block_h;
        if (row_hi > rows) {
            row_hi = rows;
        }
        if (row_lo == 0) {
            cb_reserve_back(cb_activation_block, act_block_tiles);
        }
        if (row_lo >= row_hi) {
            return;
        }
        if (x_mcast.should_send(x_round_offset + row_block_idx)) {
            const uint32_t slot_addr = get_write_ptr(cb_activation_block);
            tag_reads(x_trid);
            const uint8_t read_noc = row_block_idx == 0 ? noc_index : x_late_noc;
            if (read_noc != noc_index) {
                noc_async_read_set_trid(x_late_trid, read_noc);
            }
            for (uint32_t row = row_lo; row < row_hi; ++row) {
                const uint32_t x_page = (row_start + first_row + row) * k_tiles;
                const uint32_t row_addr = slot_addr + row * k_ext_tiles * tile_bytes;
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    noc_async_read_page(x_page + k, x_accessor, row_addr + k * tile_bytes, 0, read_noc);
                }
            }
            if (read_noc != noc_index) {
                noc_async_read_set_trid(0, read_noc);
            }
            if (row_hi == rows) {
                dataflow_kernel_lib::set_read_state<tile_bytes>(noc, ones_addr);
                for (uint32_t row = 0; row < rows; ++row) {
                    dataflow_kernel_lib::read_with_state<tile_bytes>(
                        noc, slot_addr + (row * k_ext_tiles + k_tiles) * tile_bytes, ones_addr);
                }
            }
            tag_reads(0);
        }
    };
    auto issue_activation_reads = [&](uint32_t row_block_idx) { issue_activation_rows(row_block_idx, 0, block_h); };
    auto finish_activation_block = [&](uint32_t row_block_idx) {
        MaybeDeviceZoneScope("reader_x_mcast");
        const uint32_t first_row = row_block_idx * block_h;
        const uint32_t rows = (core_rows - first_row) < block_h ? (core_rows - first_row) : block_h;
        const uint32_t slot_addr = get_write_ptr(cb_activation_block);
        if (x_mcast.should_send(x_round_offset + row_block_idx)) {
            wait_reads(x_trid);
            if (row_block_idx != 0 && x_late_noc != noc_index) {
                noc_async_read_barrier_with_trid(x_late_trid, x_late_noc);
            }
            x_sender->send(slot_addr, slot_addr, rows * k_ext_tiles * tile_bytes);
        } else {
            x_receiver->receive(x_round_offset + row_block_idx);
        }
        // Nominal quantum keeps the ring aligned on the tail; rows past `rows` are never consumed.
        cb_push_back(cb_activation_block, act_block_tiles);
    };

    // Built once: the pipes carry per-round state. A one-row column has no receivers, so either may be absent.
    auto w_sender = w_mcast.optional_sender(noc);
    auto w_receiver = w_mcast.optional_receiver(noc);
    // The slice is never popped and the CB holds all of it, so a chunk's slot is free before its reserve_back.
    const uint32_t weight_base = weight_cb.get_write_ptr();
    auto read_weight_chunk = [&](uint32_t n_block_idx) {
        const uint32_t chunk_addr = weight_base + n_block_idx * w_chunk_tiles * tile_bytes;
        const uint32_t n_first = n_tile_start + n_block_idx * block_w;
        tag_reads(w_trid);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t w = 0; w < block_w; ++w) {
                noc_async_read_page(k * n_tiles + n_first + w, w_accessor, chunk_addr + (k * block_w + w) * tile_bytes);
            }
        }
        for (uint32_t w = 0; w < block_w; ++w) {
            noc_async_read_page(n_first + w, bias_accessor, chunk_addr + (k_tiles * block_w + w) * tile_bytes);
        }
        tag_reads(0);
    };
    auto issue_owned_chunk_read = [&](uint32_t n_block_idx) {
        if (n_block_idx < num_n_blocks && w_mcast.should_send(w_round_offset + n_block_idx)) {
            read_weight_chunk(n_block_idx);
        }
    };
    auto deliver_weight_block = [&](uint32_t n_block_idx) {
        if (n_block_idx >= w_prefetch_rounds) {
            issue_owned_chunk_read(n_block_idx);
        }
        weight_cb.reserve_back(w_chunk_tiles);
        const uint32_t chunk_addr = weight_cb.get_write_ptr();
        if (w_mcast.should_send(w_round_offset + n_block_idx)) {
            wait_reads(w_trid);
            // Bias tile rows 1..31 are padding: zero them so 0 * padding cannot inject NaN. Offsets are relative
            // to the CB write pointer, which is chunk_addr until the push.
            for (uint32_t w = 0; w < block_w; ++w) {
                const uint32_t tile_off = (k_tiles * block_w + w) * tile_bytes;
                noc.async_write_zeros(
                    weight_cb, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_row_bytes});
                noc.async_write_zeros(
                    weight_cb, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_bytes + face_row_bytes});
                noc.async_write_zeros(weight_cb, 2 * face_bytes, {.offset_bytes = tile_off + 2 * face_bytes});
            }
            noc.write_zeros_l1_barrier();
            w_sender->send(chunk_addr, chunk_addr, w_chunk_tiles * tile_bytes);
        } else {
            w_receiver->receive(w_round_offset + n_block_idx);
        }
        weight_cb.push_back(w_chunk_tiles);
    };

    {
        MaybeDeviceZoneScope("reader_w_prefetch");
        for (uint32_t n_block_idx = 0; n_block_idx < w_prefetch_rounds; ++n_block_idx) {
            issue_owned_chunk_read(n_block_idx);
        }
    }
    // Fill: the first w_chunks_before_x W chunks are delivered while the x round-0 sender's reads are in flight.
    // Row-block 0 computes each W chunk as it lands.
    const uint32_t num_w_first = num_n_blocks < w_chunks_before_x ? num_n_blocks : w_chunks_before_x;
    issue_activation_reads(0);
    {
        MaybeDeviceZoneScope("reader_w_first");
        for (uint32_t n_block_idx = 0; n_block_idx < num_w_first; ++n_block_idx) {
            deliver_weight_block(n_block_idx);
        }
    }
    finish_activation_block(0);
    // Row-block 1's slot is free from the start (ACT_DEPTH 2): issue its sender reads before W chunk X1_ISSUE_AT
    // and its row multicast before W chunk X1_FINISH_AT instead of after the last chunk. Every core of a grid row
    // has the same row-blocks and walks the same W chunk indices (clamped to its column's count), so the row
    // handshake and the column handshakes keep one global order and cannot deadlock.
#ifndef X1_ISSUE_AT
#define X1_ISSUE_AT 0xFFFF
#endif
#ifndef X1_FINISH_AT
#define X1_FINISH_AT 0xFFFF
#endif
    // X1_ROWS_PER_CHUNK: row-block 1's reads go out X1_ROWS_PER_CHUNK rows before each W chunk from X1_ISSUE_AT on
    // (spreads the read burst over several W rounds) instead of all at once.
#ifndef X1_ROWS_PER_CHUNK
#define X1_ROWS_PER_CHUNK block_h
#endif
    const bool has_rb1 = num_row_blocks > 1;
    // X1_ISSUE_HALF: issue before the middle W chunk (column-local count; any index is deadlock-free since the
    // reads are local to the sender and only the multicast takes part in the row handshake).
#ifdef X1_ISSUE_HALF
    const uint32_t rb1_issue_at = num_n_blocks / 2;
#else
    const uint32_t rb1_issue_at = X1_ISSUE_AT;
#endif
    bool rb1_issued = false;
    uint32_t rb1_rows_issued = 0;
    bool rb1_finished = false;
    {
        MaybeDeviceZoneScope("reader_w_deliver");
        for (uint32_t n_block_idx = num_w_first; n_block_idx < num_n_blocks; ++n_block_idx) {
            if (has_rb1 && !rb1_issued && n_block_idx >= rb1_issue_at) {
                issue_activation_rows(1, rb1_rows_issued, rb1_rows_issued + X1_ROWS_PER_CHUNK);
                rb1_rows_issued += X1_ROWS_PER_CHUNK;
                rb1_issued = rb1_rows_issued >= block_h;
            }
            if (has_rb1 && !rb1_finished && n_block_idx >= X1_FINISH_AT) {
                finish_activation_block(1);
                rb1_finished = true;
            }
            deliver_weight_block(n_block_idx);
        }
    }
    if (has_rb1) {
        if (!rb1_issued) {
            issue_activation_rows(1, rb1_rows_issued, block_h);
        }
        if (!rb1_finished) {
            finish_activation_block(1);
        }
    }
    for (uint32_t row_block_idx = 2; row_block_idx < num_row_blocks; ++row_block_idx) {
        issue_activation_reads(row_block_idx);
        finish_activation_block(row_block_idx);
    }
}
