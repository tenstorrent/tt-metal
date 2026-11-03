// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// linear_gelu reader (NCRISC, NoC0). w_delivery_pipeline candidate:
// - W_LOOKAHEAD L (define, default 0 = baseline): at the start of round r (r >= W_LA_START) the owner of chunk
//   r + L issues its DRAM read, so chunk r + L's read overlaps the multicasts of rounds r .. r + L - 1 instead of
//   starting after round r + L - 1 has landed. Rounds < W_LA_START keep the baseline issue point, so chunk 0's
//   read (and its DRAM bandwidth) is not shared with chunk 1's.
// - W reads are tagged with a per-chunk transaction id (W_TRID): the owner's barrier waits only for its round's
//   chunk, not for later prefetched chunks (a core owns several chunks when the column has fewer rows than chunks).
// - W_GROUP G: rounds after round 0 multicast G chunks at once (fewer handshakes, bigger payloads).
// - W_NOHS (with the host's W channel handshake=False + Counter): the receiver waits on the relative round.
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
#ifdef W_SHIFT_M1
    // Host shift 1 -> effective shift -1: chunk 1's owner in column c is row c, the x round-0 sender of its row.
    const uint32_t w_round_offset = get_arg_val<uint32_t>(8) + (w_mcast.num_senders >= 2 ? w_mcast.num_senders - 2 : 0);
#else
    const uint32_t w_round_offset = get_arg_val<uint32_t>(8);
#endif

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
    // X_TRID: x reads carry trid 15 (W reads cycle 1..14), so the x sender's barrier skips W reads in flight.
    constexpr uint32_t x_trid = 15;

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
#ifdef X_TRID
            noc_async_read_set_trid(x_trid);
#endif
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
#ifdef X_TRID
            noc_async_read_set_trid(0);
#endif
        }
    };
    auto finish_activation_block = [&](uint32_t row_block_idx) {
        MaybeDeviceZoneScope("reader_x_mcast");
        const uint32_t first_row = row_block_idx * block_h;
        const uint32_t rows = (core_rows - first_row) < block_h ? (core_rows - first_row) : block_h;
        const uint32_t slot_addr = get_write_ptr(cb_activation_block);
        if (x_mcast.should_send(x_round_offset + row_block_idx)) {
#ifdef X_TRID
            noc_async_read_barrier_with_trid(x_trid);  // x reads only; W reads in flight carry their own trids
#else
            noc_async_read_barrier();
#endif
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
#ifndef W_READ_NOC
#define W_READ_NOC 0
#endif
    // W_READ_NOC 1 (needs W_TRID and DM_DYNAMIC_NOC on both DM kernels): W read responses travel on NoC1's links,
    // off the NoC0 column links that carry the W and x multicasts.
    constexpr uint8_t w_noc = W_READ_NOC;
    // K tile rows [k_begin, k_end) of chunk n_block_idx; row k_tiles is the bias tile row.
    auto read_weight_rows = [&](uint32_t n_block_idx, uint32_t k_begin, uint32_t k_end) {
        const uint32_t chunk_addr = weight_base + n_block_idx * w_chunk_tiles * tile_bytes;
        const uint32_t n_first = n_tile_start + n_block_idx * block_w;
        const uint32_t w_end = k_end < k_tiles ? k_end : k_tiles;
        for (uint32_t k = k_begin; k < w_end; ++k) {
            for (uint32_t w = 0; w < block_w; ++w) {
                noc_async_read_page(
                    k * n_tiles + n_first + w, w_accessor, chunk_addr + (k * block_w + w) * tile_bytes, 0, w_noc);
            }
        }
        if (k_end <= k_tiles) {
            return;
        }
        for (uint32_t w = 0; w < block_w; ++w) {
            const uint32_t bias_dst = chunk_addr + (k_tiles * block_w + w) * tile_bytes;
#ifdef W_BIAS_ROW
            // Only row 0 of faces 0 and 1 is bias data; the padding was zeroed up front (zero_owned_bias_padding).
            noc_async_read(bias_accessor.get_noc_addr(n_first + w, 0, w_noc), bias_dst, face_row_bytes, w_noc);
            noc_async_read(
                bias_accessor.get_noc_addr(n_first + w, face_bytes, w_noc),
                bias_dst + face_bytes,
                face_row_bytes,
                w_noc);
#else
            noc_async_read_page(n_first + w, bias_accessor, bias_dst, 0, w_noc);
#endif
        }
    };
    auto read_weight_chunk = [&](uint32_t n_block_idx) { read_weight_rows(n_block_idx, 0, k_ext_tiles); };
    // Bias tile rows 1..31 are padding: zero them so 0 * padding cannot inject NaN. offset is relative to the
    // CB write pointer.
    auto zero_bias_padding = [&](uint32_t offset, uint32_t chunks) {
        for (uint32_t c = 0; c < chunks; ++c) {
            for (uint32_t w = 0; w < block_w; ++w) {
                const uint32_t tile_off = offset + (c * w_chunk_tiles + k_tiles * block_w + w) * tile_bytes;
                noc.async_write_zeros(
                    weight_cb, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_row_bytes});
                noc.async_write_zeros(
                    weight_cb, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_bytes + face_row_bytes});
                noc.async_write_zeros(weight_cb, 2 * face_bytes, {.offset_bytes = tile_off + 2 * face_bytes});
            }
        }
        noc.write_zeros_l1_barrier();
    };
#ifndef W_LOOKAHEAD
#define W_LOOKAHEAD 0
#endif
#ifndef W_LA_START
#define W_LA_START 1
#endif
#ifndef W_GROUP
#define W_GROUP 1
#endif
    constexpr uint32_t w_lookahead = W_LOOKAHEAD;
    constexpr uint32_t w_la_start = W_LA_START;
    // Round 0 carries chunk 0 alone (it gates the fill); round g >= 1 carries W_GROUP chunks in one multicast.
    constexpr uint32_t w_group = W_GROUP;
    auto round_first = [](uint32_t round) -> uint32_t { return round == 0 ? 0 : 1 + (round - 1) * w_group; };
    auto round_end = [&](uint32_t round) -> uint32_t {
        const uint32_t end = round == 0 ? 1 : 1 + round * w_group;
        return end < num_n_blocks ? end : num_n_blocks;
    };
    const uint32_t num_w_rounds = num_n_blocks == 0 ? 0 : 1 + (num_n_blocks - 1 + w_group - 1) / w_group;
    // trids 1..14 cycle over rounds; x reads use trid 0 (or 15 with X_TRID).
    auto w_trid = [](uint32_t round) -> uint32_t { return 1 + round % 14; };
    uint32_t w_issued = 0;  // rounds [0, w_issued) have had their (owned) reads issued
    auto issue_owned_round_read = [&](uint32_t round) {
        if (w_mcast.should_send(w_round_offset + round)) {
#ifdef W_TRID
            noc_async_read_set_trid(w_trid(round), w_noc);
#endif
            for (uint32_t j = round_first(round); j < round_end(round); ++j) {
                read_weight_chunk(j);
            }
#ifdef W_TRID
            noc_async_read_set_trid(0, w_noc);
#endif
        }
    };
    auto issue_owned_up_to = [&](uint32_t end) {
        end = end < num_w_rounds ? end : num_w_rounds;
        for (; w_issued < end; ++w_issued) {
            issue_owned_round_read(w_issued);
        }
    };
#ifdef W_READ_SPLIT
    // Chunk c's read is paced over the P rounds before it: part p at the start of round c - P + p. A reading core
    // then ingests 1/P of a chunk per round next to the round's multicast instead of a whole chunk at once.
    constexpr uint32_t w_split = W_READ_SPLIT;
    static_assert(w_group == 1, "W_READ_SPLIT assumes one chunk per round");
    constexpr uint32_t max_chunks = 128;
    uint8_t parts_done[max_chunks] = {};
    auto issue_parts = [&](uint32_t c, uint32_t target) {
        if (c >= num_n_blocks || parts_done[c] >= target || !w_mcast.should_send(w_round_offset + c)) {
            return;
        }
#ifdef W_TRID
        noc_async_read_set_trid(w_trid(c), w_noc);
#endif
        read_weight_rows(c, parts_done[c] * k_ext_tiles / w_split, target * k_ext_tiles / w_split);
#ifdef W_TRID
        noc_async_read_set_trid(0, w_noc);
#endif
        parts_done[c] = target;
    };
    auto issue_split = [&](uint32_t round) {
        issue_parts(round, w_split);
        if (round < w_la_start) {
            return;
        }
        for (uint32_t c = round + 1; c <= round + w_split; ++c) {
            issue_parts(c, round + w_split + 1 - c);
        }
    };
#endif
    auto deliver_weight_round = [&](uint32_t round) {
#ifdef W_ROUND_ZONES
        MaybeDeviceZoneScope("w_round");
#endif
        const uint32_t issue_end = round + 1 + (round >= w_la_start ? w_lookahead : 0);
        const bool sends = w_mcast.should_send(w_round_offset + round);
#if defined(W_READ_SPLIT)
        issue_split(round);
#elif defined(W_ISSUE_AFTER_SEND)
        // The round's sender issues its lookahead reads after its multicast is under way (it is the one core of
        // the column that does nothing while the round lands); receivers issue theirs before waiting.
        if (!sends) {
            issue_owned_up_to(issue_end);
        } else {
            issue_owned_up_to(round + 1);
        }
#else
        issue_owned_up_to(issue_end);
#endif
        const uint32_t chunks = round_end(round) - round_first(round);
        weight_cb.reserve_back(chunks * w_chunk_tiles);
        const uint32_t chunk_addr = weight_cb.get_write_ptr();
        if (sends) {
            {
#ifdef W_ROUND_ZONES
                MaybeDeviceZoneScope("w_rdbar");
#endif
#ifdef W_TRID
                noc_async_read_barrier_with_trid(w_trid(round), w_noc);
#else
                static_assert(W_READ_NOC == 0, "W_READ_NOC 1 needs W_TRID");
                noc_async_read_barrier();
#endif
            }
#if !defined(W_BIAS_ROW) && !defined(W_DEBUG_NO_ZERO)
            zero_bias_padding(0, chunks);
#endif
#ifdef W_ROUND_ZONES
            MaybeDeviceZoneScope("w_send");
#endif
#ifdef W_SEND_NOGUARD
            // The source is the resident slice, never overwritten: no need to wait for the NIU to read it.
            w_sender->template send<dataflow_kernel_lib::SourceL1Guard::CallerManaged>(
                chunk_addr, chunk_addr, chunks * w_chunk_tiles * tile_bytes);
#else
            w_sender->send(chunk_addr, chunk_addr, chunks * w_chunk_tiles * tile_bytes);
#endif
#ifdef W_ISSUE_AFTER_SEND
            issue_owned_up_to(issue_end);
#endif
        } else {
#ifdef W_ROUND_ZONES
            MaybeDeviceZoneScope("w_recv");
#endif
#ifdef W_NOHS
            w_receiver->receive(round);
#else
            w_receiver->receive(w_round_offset + round);
#endif
        }
        weight_cb.push_back(chunks * w_chunk_tiles);
    };

    {
        MaybeDeviceZoneScope("reader_w_prefetch");
#ifdef W_READ_SPLIT
        issue_parts(0, w_split);
#else
        issue_owned_up_to(w_prefetch_rounds);
#endif
#ifdef W_BIAS_ROW
        // Every slot is free from the start (the CB is never popped), so owners zero all their bias padding now.
        for (uint32_t round = 0; round < num_w_rounds; ++round) {
            if (w_mcast.should_send(w_round_offset + round)) {
                zero_bias_padding(
                    round_first(round) * w_chunk_tiles * tile_bytes, round_end(round) - round_first(round));
            }
        }
#endif
    }
    // Fill: the first w_chunks_before_x W rounds (round 0 = chunk 0) are delivered while the x round-0 sender's
    // reads are in flight. Row-block 0 computes each W chunk as it lands; row-block 1 is not needed until all are
    // consumed.
    const uint32_t num_w_first = num_w_rounds < w_chunks_before_x ? num_w_rounds : w_chunks_before_x;
#ifndef W_EARLY_ON_XSEND
#define W_EARLY_ON_XSEND 0
#endif
    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        issue_activation_reads(row_block_idx);
        if constexpr (W_EARLY_ON_XSEND > 0) {
            // The x round-0 sender ingests nothing while it multicasts row-block 0: its owned W reads of rounds
            // 1 .. W_EARLY_ON_XSEND land in that window instead of next to a later W multicast.
            if (row_block_idx == 0 && x_mcast.should_send(x_round_offset)) {
                noc_async_read_barrier_with_trid(x_trid);
                for (uint32_t round = 1; round <= W_EARLY_ON_XSEND && round < num_w_rounds; ++round) {
#ifdef W_READ_SPLIT
                    issue_parts(round, w_split);
#else
                    if (w_issued == round) {
                        issue_owned_up_to(round + 1);
                    }
#endif
                }
            }
        }
        if (row_block_idx == 0) {
            MaybeDeviceZoneScope("reader_w_first");
            for (uint32_t round = 0; round < num_w_first; ++round) {
                deliver_weight_round(round);
            }
        }
        finish_activation_block(row_block_idx);
        if (row_block_idx == 0) {
            MaybeDeviceZoneScope("reader_w_deliver");
            for (uint32_t round = num_w_first; round < num_w_rounds; ++round) {
                deliver_weight_round(round);
            }
        }
    }
}
