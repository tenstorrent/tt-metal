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

// fill_first_block experiment: W channel handshake=False + Counter data-ready (cb_weight_resident is never
// popped, so no slot can be overwritten early; dispatch initializes semaphores before any core starts).
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
    auto issue_activation_reads = [&](uint32_t row_block_idx) {
        MaybeDeviceZoneScope("reader_x_issue");
        const uint32_t first_row = row_block_idx * block_h;
        const uint32_t rows = (core_rows - first_row) < block_h ? (core_rows - first_row) : block_h;
        cb_reserve_back(cb_activation_block, act_block_tiles);
        if (x_mcast.should_send(x_round_offset + row_block_idx)) {
            const uint32_t slot_addr = get_write_ptr(cb_activation_block);
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t x_page = (row_start + first_row + row) * k_tiles;
                const uint32_t row_addr = slot_addr + row * k_ext_tiles * tile_bytes;
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    noc_async_read_page(x_page + k, x_accessor, row_addr + k * tile_bytes);
                }
            }
            dataflow_kernel_lib::set_read_state<tile_bytes>(noc, ones_addr);
            for (uint32_t row = 0; row < rows; ++row) {
                dataflow_kernel_lib::read_with_state<tile_bytes>(
                    noc, slot_addr + (row * k_ext_tiles + k_tiles) * tile_bytes, ones_addr);
            }
        }
    };
    auto finish_activation_block = [&](uint32_t row_block_idx) {
        MaybeDeviceZoneScope("reader_x_mcast");
        const uint32_t first_row = row_block_idx * block_h;
        const uint32_t rows = (core_rows - first_row) < block_h ? (core_rows - first_row) : block_h;
        const uint32_t slot_addr = get_write_ptr(cb_activation_block);
        if (x_mcast.should_send(x_round_offset + row_block_idx)) {
            noc_async_read_barrier();
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
        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t w = 0; w < block_w; ++w) {
                noc_async_read_page(k * n_tiles + n_first + w, w_accessor, chunk_addr + (k * block_w + w) * tile_bytes);
            }
        }
        for (uint32_t w = 0; w < block_w; ++w) {
            noc_async_read_page(n_first + w, bias_accessor, chunk_addr + (k_tiles * block_w + w) * tile_bytes);
        }
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
            noc_async_read_barrier();
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
            // No handshake on the W channel (Counter data-ready): the receiver waits on the RELATIVE round, the
            // count of chunks delivered so far; the absolute round only picks the owner (should_send above).
            w_receiver->receive(n_block_idx);
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
    // Row-block 0 computes each W chunk as it lands; row-block 1 is not needed until all are consumed.
#ifdef W0_OWNER_FIRST
    // Without a W handshake nobody waits on the chunk-0 owner's x round, so the owner multicasts chunk 0 before its
    // own x row-block 0 and the two first multicasts overlap; everyone else keeps x first.
    const uint32_t num_w_first = (num_n_blocks > 0 && w_mcast.should_send(w_round_offset)) ? 1 : 0;
#else
    const uint32_t num_w_first = num_n_blocks < w_chunks_before_x ? num_n_blocks : w_chunks_before_x;
#endif
    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        issue_activation_reads(row_block_idx);
        if (row_block_idx == 0) {
            MaybeDeviceZoneScope("reader_w_first");
            for (uint32_t n_block_idx = 0; n_block_idx < num_w_first; ++n_block_idx) {
                deliver_weight_block(n_block_idx);
            }
        }
        finish_activation_block(row_block_idx);
        if (row_block_idx == 0) {
            MaybeDeviceZoneScope("reader_w_deliver");
            for (uint32_t n_block_idx = num_w_first; n_block_idx < num_n_blocks; ++n_block_idx) {
                deliver_weight_block(n_block_idx);
            }
        }
    }
}
