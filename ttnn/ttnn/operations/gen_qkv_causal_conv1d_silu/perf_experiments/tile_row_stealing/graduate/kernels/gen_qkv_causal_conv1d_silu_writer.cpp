// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu writer (BRISC / NoC1) — store_block.
//
// Global channel tile cg routes to q if cg < Q_t, to k if cg < Q_t + K_t, else to v (page r * X_t + (cg - base_X)).
// Compute pushes each tile-row in groups of out_group_tiles; per group: wait, issue its writes, flush, pop.
// Split reader (ROW_MAJOR regime, split_rows > 0; Refinement 2): this RISC also reads body rows
// [32 - split_rows, 32) of every input tile-row on NoC1 into the reader's slot (cb_split_requests ->
// reads -> cb_split_done).
// Static mode (CT dynamic == 0: nothing to steal — one block per core, or one core per channel group):
// the core's own contiguous token range.
// Dynamic mode (work stealing, Perf 2): the tile-rows this core stores are the ones its reader claimed (own range
// front-first, then rows stolen from the back of other ranges of the channel group). The reader posts one
// cb_split_requests page per tile-row, {L1 dst, tile-row, rows in its block, flags}, in block order, and
// a final page with flags = REQ_DONE. This kernel serves them exactly like the op's writer serves its
// split-read requests (before waiting on output block b it serves every request through the end of
// block b + 1 when it reads a split share, through block b otherwise), recording each block's first
// tile-row and row count in a small ring; output block b is stored to the tile-rows recorded for b.
// Deadlock-free for the same reason as the op's writer: the request of block b + 1 is posted once
// compute released block b - 1's slot, whose outputs this kernel has already drained; REQ_DONE is
// posted after the reader's failed claim, which follows the same slot reservation.
//
// Raw NoC writes rather than write_sticks_after_untilize: outputs are TILE pages routed to three
// tensors; that helper writes RM sticks to one tensor.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (MaybeDeviceZoneScope, PERMANENT): per served request writer_split_req_wait,
// writer_split_issue, writer_split_barrier; per output group writer_out_wait, writer_out_issue,
// writer_out_flush.
#ifdef GQKV_ABLATE_READS
#define GQKV_READ(...)
#else
#define GQKV_READ(...) noc_async_read(__VA_ARGS__)
#endif
#ifdef GQKV_ABLATE_WRITES
#define GQKV_WRITE(...)
#else
#define GQKV_WRITE(...) noc_async_write(__VA_ARGS__)
#endif

namespace {

constexpr uint32_t cb_output_tiles = 3;
constexpr uint32_t cb_split_requests = 6;
constexpr uint32_t cb_split_done = 7;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t REQ_DONE = 1u;
constexpr uint32_t BLOCK_RING = 4;  // served blocks live at once: b and b + 1

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t split_rows = get_compile_time_arg_val(6);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_group_tiles = get_compile_time_arg_val(8);
    constexpr bool stagger_row_walk = get_compile_time_arg_val(9) == 1;
    constexpr bool dynamic = get_compile_time_arg_val(10) != 0;
    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto input_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    static_assert(split_rows <= TILE_DIM, "split_rows is a row count inside one tile-row");

    uint32_t arg = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t input_col_offset_tiles = get_arg_val<uint32_t>(arg++);

    const auto q_acc = TensorAccessor(q_args, q_addr);
    const auto k_acc = TensorAccessor(k_args, k_addr);
    const auto v_acc = TensorAccessor(v_args, v_addr);

    auto store_tile = [&](uint32_t src, uint32_t r, uint32_t cg) {
        if (cg < q_tiles) {
            GQKV_WRITE(src, q_acc.get_noc_addr(r * q_tiles + cg), tile_bytes);
        } else if (cg < q_tiles + k_tiles) {
            GQKV_WRITE(src, k_acc.get_noc_addr(r * k_tiles + (cg - q_tiles)), tile_bytes);
        } else {
            GQKV_WRITE(src, v_acc.get_noc_addr(r * v_tiles + (cg - q_tiles - k_tiles)), tile_bytes);
        }
    };
    auto store_row = [&](uint32_t r) {
        for (uint32_t c0 = 0; c0 < block_ch_tiles; c0 += out_group_tiles) {
            const uint32_t group = (block_ch_tiles - c0 < out_group_tiles) ? (block_ch_tiles - c0) : out_group_tiles;
            {
                MaybeDeviceZoneScope("writer_out_wait");
                cb_wait_front(cb_output_tiles, group);
            }
            const uint32_t src = get_read_ptr(cb_output_tiles);
            {
                MaybeDeviceZoneScope("writer_out_issue");
                for (uint32_t i = 0; i < group; ++i) {
                    store_tile(src + i * tile_bytes, r, ch_group_tile_start + c0 + i);
                }
            }
            {
                MaybeDeviceZoneScope("writer_out_flush");
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_output_tiles, group);
        }
    };
    // Split share of tile-row `row` (token rows [32 - split_rows, 32)) into the reader's slot at dst.
    [[maybe_unused]] auto split_read_row = [&](uint32_t row, uint32_t dst) {
        if constexpr (split_rows > 0) {
            const auto input_acc = TensorAccessor(input_args, input_addr);
            const uint32_t input_byte_offset =
                (input_col_offset_tiles + ch_group_tile_start) * (stick_bytes / block_ch_tiles);
            const uint32_t token0 = row * TILE_DIM;
            {
                MaybeDeviceZoneScope("writer_split_issue");
                for (uint32_t j = 0; j < split_rows; ++j) {
                    const uint32_t i = TILE_DIM - split_rows +
                                       (stagger_row_walk ? (j + ch_group_tile_start + core_seq_start) % split_rows : j);
                    GQKV_READ(
                        input_acc.get_noc_addr(token0 + i, input_byte_offset), dst + i * stick_bytes, stick_bytes);
                }
            }
            {
                MaybeDeviceZoneScope("writer_split_barrier");
                noc_async_read_barrier();
            }
            cb_reserve_back(cb_split_done, 1);
            cb_push_back(cb_split_done, 1);
        }
    };

    if constexpr (!dynamic) {
        // ---- the op's writer ----
        [[maybe_unused]] uint32_t rows_served = 0;
        [[maybe_unused]] auto serve_split_rows_through = [&](uint32_t end) {
            if constexpr (split_rows > 0) {
                for (; rows_served < end; ++rows_served) {
                    {
                        MaybeDeviceZoneScope("writer_split_req_wait");
                        cb_wait_front(cb_split_requests, 1);
                    }
                    invalidate_l1_cache();
                    const uint32_t dst =
                        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_split_requests));
                    cb_pop_front(cb_split_requests, 1);
                    split_read_row(core_seq_start + rows_served, dst);
                }
            }
        };
        for (uint32_t r = core_seq_start; r < core_seq_start + core_seq_tiles; ++r) {
            if constexpr (split_rows > 0) {
                const uint32_t m = r - core_seq_start;
                const uint32_t next_block_end = (m / block_rows + 2) * block_rows;
                serve_split_rows_through(next_block_end < core_seq_tiles ? next_block_end : core_seq_tiles);
            }
            store_row(r);
        }
    } else {
        // ---- dynamic: rows as claimed by the reader ----
        uint32_t ring_first[BLOCK_RING];
        uint32_t ring_rows[BLOCK_RING];
        uint32_t blocks_seen = 0;  // blocks whose first request has been served
        uint32_t rows_left = 0;    // requests still to serve in the newest seen block
        bool done = false;
        auto serve_one = [&]() {
            {
                MaybeDeviceZoneScope("writer_split_req_wait");
                cb_wait_front(cb_split_requests, 1);
            }
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* page =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_split_requests));
            const uint32_t dst = page[0];
            const uint32_t row = page[1];
            const uint32_t rows = page[2];
            const uint32_t flags = page[3];
            cb_pop_front(cb_split_requests, 1);
            if (flags & REQ_DONE) {
                done = true;
                return;
            }
            if (rows_left == 0) {
                ring_first[blocks_seen % BLOCK_RING] = row;
                ring_rows[blocks_seen % BLOCK_RING] = rows;
                ++blocks_seen;
                rows_left = rows;
            }
            --rows_left;
            split_read_row(row, dst);
        };
        constexpr uint32_t lookahead = split_rows > 0 ? 2 : 1;  // blocks served before storing block b
        for (uint32_t b = 0;; ++b) {
            while (!done && blocks_seen - (rows_left > 0 ? 1 : 0) < b + lookahead) {
                serve_one();
            }
            if (b >= blocks_seen) {
                break;  // REQ_DONE seen and every block stored
            }
            const uint32_t first = ring_first[b % BLOCK_RING];
            const uint32_t rows = ring_rows[b % BLOCK_RING];
            for (uint32_t m = 0; m < rows; ++m) {
                store_row(first + m);
            }
        }
    }
    noc_async_write_barrier();
}
