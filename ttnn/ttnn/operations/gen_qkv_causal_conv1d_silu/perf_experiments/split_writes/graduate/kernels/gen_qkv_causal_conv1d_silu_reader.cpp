// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu reader (NCRISC).
//
// Block operations owned here (op_design.md -> Block schedule):
//   load_taps        once per channel group (regime tap_multicast_per_group): the group's sender core
//                    (slot 0) reads row 0 of tap p's tile into row 2p of the packed tap tile (2 x 32 B
//                    reads per (tap, channel tile)) and multicasts the packed tiles to the other cores
//                    of the group (kernel_lib mcast pipe). Every other core issues its first tile-row's
//                    input reads, then receives the taps. Taps cross DRAM once per channel instead of
//                    once per core: at 110 cores the per-core tap fetch (~14k tiny DRAM requests)
//                    measured up to 80 us ahead of the first input row.
//   load_rows_block  one cb_rows slot per block: sticks [0,3) = halo (tokens t0-3..t0-1),
//                    sticks [3, 3 + 32*rows) = body. A stick is the block's contiguous channel slice
//                    of one token row (64 * block_ch_tiles B).
//                    halo source: t0 == 0          -> history rows 0..2 (read_history_halo)
//                                 later block      -> previous slot's last 3 body sticks (L1 -> L1)
//                                 core-first, t0>0 -> RM: input rows t0-3..t0-1;
//                                                     TILE: rows 29..31 of untilized tile-row t0/32-1
//                    RM:   per tile-row 32 stick reads (+ halo on the first) -> ONE barrier -> push.
//                          Split reader (split_rows > 0): body rows [32 - split_rows, 32) of every
//                          tile-row are read by the writer RISC on the other NoC, whose DRAM
//                          endpoints are separate ports: decode is DRAM-request-rate-bound per port
//                          (Refinement 2: ~12.5 ns per request per bank port). Per tile-row this
//                          kernel posts the row's L1 destination to cb_split_requests, reads its own
//                          rows, then waits for cb_split_done before pushing the rows.
//                    TILE: per tile-row: issue the NEXT input tile-row's reads, then copy the
//                          untilized tile-row (32 contiguous sticks) into the slot -> ONE barrier.
//
// store_share  (split writes, write_share > 0): this RISC also writes the q / k / v tiles (r, cg) with
//              ((r + cg) * write_share) % 8 < write_share on its NoC (NoC0), so the DRAM writes use both
//              NoCs' link sets. cb_output_tiles keeps ONE consumer (the writer pops): this kernel never
//              pops it — it shadow-tracks the CB's tiles_received counter (reset to 0 at launch) and its
//              own copy of the read pointer, writes its tiles of each ready group straight from L1,
//              flushes them (source read out) and pushes one token into cb_write_done; the writer pops
//              the group only after that token. Every blocking wait below is a polling loop that serves
//              ready groups (serve_writes), so the writer's wait on a token never deadlocks against a
//              reader wait; the rest is drained after the last input row.
//
// Raw NoC calls rather than read_sticks_for_tilize (tilize_helpers_dataflow.inl:146-157): that
// helper barriers after every stick and has no notion of a 3-row halo prefix from another source
// (history / previous slot); gap closer = a per-block batched-barrier mode with a caller prefix.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (MaybeDeviceZoneScope, PERMANENT — perf tournaments rank stages from them; free when the
// profiler is off): reader_taps_issue (group sender, once), per block reader_rows_reserve (back-pressure
// from compute), reader_halo_issue; per tile-row reader_rows_issue, reader_rows_barrier (incl. the tap
// publish on the first row), reader_split_wait (the writer's NoC1 share); TILE regime reader_tile_issue,
// reader_untilized_wait, reader_slot_copy.
// Measurement-only ablation (compiled out unless defined; wrong results by construction):
// GQKV_ABLATE_READS drops the input / history / halo DRAM reads (taps, CB sync and barriers stay).
#ifdef GQKV_ABLATE_READS
#define GQKV_READ(...)
#else
#define GQKV_READ(...) noc_async_read(__VA_ARGS__)
#endif

namespace {

constexpr uint32_t cb_rows = 0;
constexpr uint32_t cb_tap_tiles = 1;
constexpr uint32_t cb_input_tiles = 4;
constexpr uint32_t cb_untilized_rows = 5;
constexpr uint32_t cb_split_requests = 6;  // split reader: this kernel -> writer, one L1 row address per tile-row
constexpr uint32_t cb_split_done = 7;      // split reader: writer -> this kernel, one token per tile-row
constexpr uint32_t cb_output_tiles = 3;    // split writes: read-only here (the writer is the consumer)
constexpr uint32_t cb_write_done = 8;      // split writes: this kernel -> writer, one token per group written into
constexpr uint32_t WRITE_SHARE_PERIOD = 8;

// Split writes: this RISC writes write_share of every WRITE_SHARE_PERIOD tiles (spread over r + cg).
FORCE_INLINE bool reader_writes_tile(uint32_t r, uint32_t cg, uint32_t write_share) {
    return ((r + cg) * write_share) % WRITE_SHARE_PERIOD < write_share;
}

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t TAP_COUNT = 4;
constexpr uint32_t HALO_ROWS = TAP_COUNT - 1;
constexpr uint32_t TAP_ROW_STRIDE = 2;  // tap p lives in row 2p of the packed tap tile (64 B aligned)

// Self-aimed L1 -> L1 copy (dataflow_kernel_lib::local_addr: the source is this core's own L1).
FORCE_INLINE void copy_local(const Noc& noc, uint32_t src_addr, uint32_t dst_addr, uint32_t size_bytes) {
    UnicastEndpoint self_ep;
    noc.async_read(
        self_ep,
        CoreLocalMem<uint32_t>(dst_addr),
        size_bytes,
        dataflow_kernel_lib::local_addr(src_addr, noc.get_noc_id()),
        {});
}

// Row 0 of tap tile (0, ch_group_tile_start + c) -> row 2*tap_idx of packed tile c.
template <uint32_t block_ch_tiles, uint32_t tile_bytes, uint32_t elem_bytes, typename Accessor>
FORCE_INLINE void issue_tap_rows(
    const Accessor& tap_acc, uint32_t tap_idx, uint32_t tap_base, uint32_t ch_group_tile_start) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    const uint32_t row_offset = TAP_ROW_STRIDE * tap_idx * face_row_bytes;
    for (uint32_t c = 0; c < block_ch_tiles; ++c) {
        const uint64_t src = tap_acc.get_noc_addr(ch_group_tile_start + c);
        const uint32_t dst = tap_base + c * tile_bytes + row_offset;
        noc_async_read(src, dst, face_row_bytes);                            // face 0: channels 0..15
        noc_async_read(src + face_bytes, dst + face_bytes, face_row_bytes);  // face 1: channels 16..31
    }
}

// History halo: the 3 history rows of channels [c0, c0 + 32 * block_ch_tiles) -> sticks 0..2 of the slot.
//   token-paged   (history_page_channels == 0): [1, 3, C]      row r = page r, channel c at element c.
//   channel-paged (history_page_channels == P): [1, C/P, 3P]   row r of channel c = page c / P,
//                 element r * P + c % P. A group whose channel slice straddles pages issues one read
//                 per contiguous run per row; every run starts on a multiple of 32 channels (c0 and P
//                 are tile multiples), i.e. 64 B aligned in source and destination.
template <uint32_t block_ch_tiles, uint32_t history_page_channels, uint32_t elem_bytes, typename Accessor>
FORCE_INLINE void read_history_halo(const Accessor& history_acc, uint32_t slot_base, uint32_t ch_group_tile_start) {
    constexpr uint32_t group_channels = block_ch_tiles * TILE_DIM;
    constexpr uint32_t stick_bytes = group_channels * elem_bytes;
    const uint32_t c0 = ch_group_tile_start * TILE_DIM;
    for (uint32_t r = 0; r < HALO_ROWS; ++r) {
        const uint32_t dst_row = slot_base + r * stick_bytes;
        if constexpr (history_page_channels == 0) {
            GQKV_READ(history_acc.get_noc_addr(r, c0 * elem_bytes), dst_row, stick_bytes);
        } else {
            constexpr uint32_t P = history_page_channels;
            for (uint32_t c = c0; c < c0 + group_channels;) {
                const uint32_t page = c / P;
                const uint32_t in_page = c - page * P;
                const uint32_t run_end = (page + 1) * P < c0 + group_channels ? (page + 1) * P : c0 + group_channels;
                GQKV_READ(
                    history_acc.get_noc_addr(page, (r * P + in_page) * elem_bytes),
                    dst_row + (c - c0) * elem_bytes,
                    (run_end - c) * elem_bytes);
                c = run_end;
            }
        }
    }
}

}  // namespace

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr bool tile_input = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t history_page_channels = get_compile_time_arg_val(6);  // 0: token-paged history
    constexpr uint32_t split_rows = get_compile_time_arg_val(7);             // body rows per tile-row the writer reads
    constexpr bool stagger_row_walk = get_compile_time_arg_val(8) == 1;      // rotate each core's row order
    constexpr auto input_args = TensorAccessorArgs<9>();
    constexpr auto history_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr uint32_t q_tiles = get_compile_time_arg_val(v_args.next_compile_time_args_offset());
    constexpr uint32_t k_tiles = get_compile_time_arg_val(v_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(v_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t out_group_tiles = get_compile_time_arg_val(v_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t write_share = get_compile_time_arg_val(v_args.next_compile_time_args_offset() + 4);
    constexpr auto taps_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("taps_mcast_ct_offset"),
        get_named_compile_time_arg_val("taps_mcast_rt_offset")>();

    constexpr uint32_t channel_tile_bytes = TILE_DIM * elem_bytes;  // one channel tile of one token row
    constexpr uint32_t stick_bytes = block_ch_tiles * channel_tile_bytes;
    constexpr uint32_t untilized_row_bytes = TILE_DIM * stick_bytes;  // one untilized tile-row = 32 sticks
    constexpr uint32_t first_row_push = TILE_DIM + HALO_ROWS;         // halo + first tile-row
    constexpr uint32_t tap_tiles_bytes = block_ch_tiles * tile_bytes;
    constexpr uint32_t own_rows = TILE_DIM - split_rows;  // body rows [0, own_rows) are read here
    static_assert(split_rows <= TILE_DIM, "split_rows is a row count inside one tile-row");
    static_assert(split_rows == 0 || !tile_input, "the split reader is a ROW_MAJOR-regime lever");

    // ---- runtime args ----
    uint32_t arg = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t history_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t input_col_offset_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);

    // ---- store_share: this RISC's q / k / v stores (shadow consumer of cb_output_tiles) ----
    [[maybe_unused]] const auto q_acc = TensorAccessor(q_args, q_addr);
    [[maybe_unused]] const auto k_acc = TensorAccessor(k_args, k_addr);
    [[maybe_unused]] const auto v_acc = TensorAccessor(v_args, v_addr);
    const uint32_t wr_end = core_seq_start + core_seq_tiles;
    uint32_t wr_row = core_seq_start;  // cursor: next output group (tile-row, first channel tile in the row)
    uint32_t wr_c0 = 0;
    uint16_t wr_seen = 0;  // cb_output_tiles tiles pushed before the cursor's group
    // Serve every ready group this RISC owns a tile of; returns at the first one not ready yet.
    auto serve_writes = [&]() {
        if constexpr (write_share > 0) {
            LocalCBInterface& out_if = get_local_cb_interface(cb_output_tiles);
            while (wr_row < wr_end) {
                const uint32_t group =
                    (block_ch_tiles - wr_c0 < out_group_tiles) ? (block_ch_tiles - wr_c0) : out_group_tiles;
                bool mine = false;
                for (uint32_t i = 0; i < group; ++i) {
                    mine |= reader_writes_tile(wr_row, ch_group_tile_start + wr_c0 + i, write_share);
                }
                if (mine) {
                    const uint16_t received = (uint16_t)reg_read((uintptr_t)get_cb_tiles_received_ptr(cb_output_tiles));
                    if ((int16_t)(uint16_t)(received - wr_seen) < (int16_t)group) {
                        return;  // compute has not pushed the group yet
                    }
                    if (!cb_pages_reservable_at_back(cb_write_done, 1)) {
                        return;
                    }
                    {
                        MaybeDeviceZoneScope("reader_out_issue");
                        const uint32_t src = out_if.fifo_rd_ptr;
                        for (uint32_t i = 0; i < group; ++i) {
                            const uint32_t cg = ch_group_tile_start + wr_c0 + i;
                            if (!reader_writes_tile(wr_row, cg, write_share)) {
                                continue;
                            }
                            const uint32_t s = src + i * tile_bytes;
                            if (cg < q_tiles) {
                                noc_async_write(s, q_acc.get_noc_addr(wr_row * q_tiles + cg), tile_bytes);
                            } else if (cg < q_tiles + k_tiles) {
                                noc_async_write(s, k_acc.get_noc_addr(wr_row * k_tiles + (cg - q_tiles)), tile_bytes);
                            } else {
                                noc_async_write(
                                    s, v_acc.get_noc_addr(wr_row * v_tiles + (cg - q_tiles - k_tiles)), tile_bytes);
                            }
                        }
                        noc_async_writes_flushed();  // L1 source read out before the writer may pop
                    }
                    cb_reserve_back(cb_write_done, 1);
                    cb_push_back(cb_write_done, 1);
                }
                // advance the shadow read pointer exactly like cb_pop_front, without touching tiles_acked
                wr_seen += group;
                out_if.fifo_rd_ptr += group * out_if.fifo_page_size;
                if (out_if.fifo_rd_ptr == out_if.fifo_limit) {
                    out_if.fifo_rd_ptr -= out_if.fifo_size;
                }
                wr_c0 += out_group_tiles;
                if (wr_c0 >= block_ch_tiles) {
                    wr_c0 = 0;
                    ++wr_row;
                }
            }
        }
    };
    // Polling replacements of the blocking waits (serve ready output groups while waiting).
    auto reserve_back_serving = [&](uint32_t cb, uint32_t n) {
        if constexpr (write_share > 0) {
            while (!cb_pages_reservable_at_back(cb, n)) {
                serve_writes();
            }
        }
        cb_reserve_back(cb, n);
    };
    auto wait_front_serving = [&](uint32_t cb, uint32_t n) {
        if constexpr (write_share > 0) {
            while (!cb_pages_available_at_front(cb, n)) {
                serve_writes();
            }
        }
        cb_wait_front(cb, n);
    };
    auto read_barrier_serving = [&]() {
        if constexpr (write_share > 0 && noc_mode != DM_DYNAMIC_NOC) {
            while (!ncrisc_noc_reads_flushed(noc_index)) {
                serve_writes();
            }
        }
        noc_async_read_barrier();
    };

    const uint32_t num_blocks_this_core = (core_seq_tiles + block_rows - 1) / block_rows;
    const uint32_t last_block_rows = core_seq_tiles - (num_blocks_this_core - 1) * block_rows;

    const auto input_acc = TensorAccessor(input_args, input_addr);
    const auto history_acc = TensorAccessor(history_args, history_addr);
    Noc noc;

    // Byte offset of this core's channel slice inside a token row of the input (column window:
    // the convolution channels start input_col_offset_tiles tiles into the row).
    const uint32_t input_byte_offset = (input_col_offset_tiles + ch_group_tile_start) * channel_tile_bytes;

    // ---- load_taps: the group's sender issues the DRAM reads; receivers wait for the multicast ----
    auto tap_sender = taps_mcast.optional_sender(noc);
    auto tap_receiver = taps_mcast.optional_receiver(noc);
    cb_reserve_back(cb_tap_tiles, block_ch_tiles);
    const uint32_t tap_base = get_write_ptr(cb_tap_tiles);  // same L1 address on every core of the group
    if (tap_sender) {
        MaybeDeviceZoneScope("reader_taps_issue");
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap0_args, tap0_addr), 0, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap1_args, tap1_addr), 1, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap2_args, tap2_addr), 2, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap3_args, tap3_addr), 3, tap_base, ch_group_tile_start);
    }
    // Called once, after the core's first tile-row reads are in flight: those reads keep landing while
    // the taps are distributed. Leaves every outstanding read complete.
    bool taps_pending = true;
    auto publish_taps_and_barrier = [&]() {
        if (tap_sender) {
            noc_async_read_barrier();  // own tap rows (and the first tile-row) landed
            tap_sender->send(tap_base, tap_base, tap_tiles_bytes);
        } else if (tap_receiver) {
            tap_receiver->receive();
        }
        noc_async_read_barrier();
        cb_push_back(cb_tap_tiles, block_ch_tiles);
        taps_pending = false;
    };

    uint32_t prev_slot_base = 0;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows_this_block = (block_idx + 1 == num_blocks_this_core) ? last_block_rows : block_rows;
        const uint32_t block_seq_start = core_seq_start + block_idx * block_rows;  // first tile-row of the block
        const uint32_t slot_sticks = TILE_DIM * rows_this_block + HALO_ROWS;

        // ---- load_rows_block: one slot ----
        {
            MaybeDeviceZoneScope("reader_rows_reserve");
            reserve_back_serving(cb_rows, slot_sticks);
        }
        const uint32_t slot_base = get_write_ptr(cb_rows);

        // Halo sticks [0, 3): tokens block_t0 - 3 .. block_t0 - 1.
        bool halo_from_untilized = false;
        {
            MaybeDeviceZoneScope("reader_halo_issue");
            if (block_seq_start == 0) {
                read_history_halo<block_ch_tiles, history_page_channels, elem_bytes>(
                    history_acc, slot_base, ch_group_tile_start);
            } else if (block_idx > 0) {
                // The previous block is full-height: its last 3 body sticks sit at [32*block_rows, +3).
                copy_local(
                    noc, prev_slot_base + TILE_DIM * block_rows * stick_bytes, slot_base, HALO_ROWS * stick_bytes);
            } else if constexpr (!tile_input) {
                const uint32_t first_token = block_seq_start * TILE_DIM - HALO_ROWS;
                for (uint32_t r = 0; r < HALO_ROWS; ++r) {
                    GQKV_READ(
                        input_acc.get_noc_addr(first_token + r, input_byte_offset),
                        slot_base + r * stick_bytes,
                        stick_bytes);
                }
            } else {
                halo_from_untilized = true;  // rows 29..31 of the extra untilized tile-row t0/32 - 1
            }
        }

        if constexpr (!tile_input) {
            // Body: per tile-row, own_rows stick reads -> ONE barrier -> [writer's rows landed] -> push.
            for (uint32_t m = 0; m < rows_this_block; ++m) {
                const uint32_t token0 = (block_seq_start + m) * TILE_DIM;
                const uint32_t dst = slot_base + (HALO_ROWS + TILE_DIM * m) * stick_bytes;
                if constexpr (split_rows > 0) {
                    cb_reserve_back(cb_split_requests, 1);
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_split_requests)) = dst;
                    cb_push_back(cb_split_requests, 1);
                }
                {
                    MaybeDeviceZoneScope("reader_rows_issue");
                    for (uint32_t j = 0; j < own_rows; ++j) {
                        const uint32_t i = stagger_row_walk ? (j + ch_group_tile_start + core_seq_start) % own_rows : j;
                        GQKV_READ(
                            input_acc.get_noc_addr(token0 + i, input_byte_offset), dst + i * stick_bytes, stick_bytes);
                    }
                }
                {
                    MaybeDeviceZoneScope("reader_rows_barrier");
                    if (taps_pending) {
                        publish_taps_and_barrier();
                    } else {
                        read_barrier_serving();
                    }
                }
                if constexpr (split_rows > 0) {
                    MaybeDeviceZoneScope("reader_split_wait");
                    wait_front_serving(cb_split_done, 1);
                    cb_pop_front(cb_split_done, 1);
                }
                cb_push_back(cb_rows, m == 0 ? first_row_push : TILE_DIM);
            }
        } else {
            // Body: input tile-rows -> compute untilize -> cb_untilized_rows -> this slot.
            const uint32_t extra_rows = halo_from_untilized ? 1 : 0;
            const uint32_t rows_to_untilize = rows_this_block + extra_rows;
            const uint32_t first_tile_row = block_seq_start - extra_rows;
            auto issue_input_tile_row = [&](uint32_t tile_row) {
                MaybeDeviceZoneScope("reader_tile_issue");
                reserve_back_serving(cb_input_tiles, block_ch_tiles);
                const uint32_t dst = get_write_ptr(cb_input_tiles);
                const uint32_t first_page = tile_row * input_width_tiles + input_col_offset_tiles + ch_group_tile_start;
                for (uint32_t c = 0; c < block_ch_tiles; ++c) {
                    GQKV_READ(input_acc.get_noc_addr(first_page + c), dst + c * tile_bytes, tile_bytes);
                }
            };

            issue_input_tile_row(first_tile_row);
            if (taps_pending) {
                publish_taps_and_barrier();
            } else {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_tiles, block_ch_tiles);
            for (uint32_t j = 0; j < rows_to_untilize; ++j) {
                const bool next_pending = (j + 1 < rows_to_untilize);
                if (next_pending) {
                    issue_input_tile_row(first_tile_row + j + 1);  // in flight while we wait on row j
                }
                {
                    MaybeDeviceZoneScope("reader_untilized_wait");
                    wait_front_serving(cb_untilized_rows, block_ch_tiles);
                }
                MaybeDeviceZoneScope("reader_slot_copy");
                const uint32_t src = get_read_ptr(cb_untilized_rows);
                if (j < extra_rows) {
                    copy_local(noc, src + (TILE_DIM - HALO_ROWS) * stick_bytes, slot_base, HALO_ROWS * stick_bytes);
                } else {
                    const uint32_t m = j - extra_rows;
                    copy_local(noc, src, slot_base + (HALO_ROWS + TILE_DIM * m) * stick_bytes, untilized_row_bytes);
                }
                noc_async_read_barrier();
                if (next_pending) {
                    cb_push_back(cb_input_tiles, block_ch_tiles);
                }
                cb_pop_front(cb_untilized_rows, block_ch_tiles);
                if (j >= extra_rows) {
                    cb_push_back(cb_rows, j == extra_rows ? first_row_push : TILE_DIM);
                }
            }
        }
        prev_slot_base = slot_base;
    }
    // store_share: every remaining output group of this core's range (the writer waits on their tokens)
    if constexpr (write_share > 0) {
        MaybeDeviceZoneScope("reader_out_drain");
        while (wr_row < wr_end) {
            serve_writes();
        }
    }
    // BB-03: nothing this kernel issued (reads, multicast writes, semaphore updates) is left in flight.
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
