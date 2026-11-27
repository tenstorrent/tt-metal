// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu writer (BRISC / NoC1) — store_block.
//
// This core owns channel tiles [ch_group_tile_start, ch_group_tile_start + block_ch_tiles) of every
// tile-row in its token range. Global channel tile cg routes to q if cg < Q_t, to k if
// cg < Q_t + K_t, else to v (page r * X_t + (cg - base_X)). Compute pushes each tile-row in groups
// of out_group_tiles (the last may be short); per group: wait, issue its writes, flush (source L1
// read out), pop — so group g is stored while compute activates group g+1. Final write barrier
// before return (BB-03).
//
// Raw NoC writes rather than write_sticks_after_untilize: outputs are TILE pages routed to three
// tensors; that helper writes RM sticks to one tensor.
//
// Split reader (ROW_MAJOR regime, split_rows > 0; Refinement 2): this RISC also reads body rows
// [32 - split_rows, 32) of every input tile-row on its own NoC (NoC1), whose DRAM endpoints are
// separate ports from the reader's (NoC0) — decode is DRAM-request-rate-bound per port. The reader
// posts each tile-row's L1 destination in cb_split_requests; this kernel issues the reads, waits for
// them, and posts a token in cb_split_done (the reader pushes the rows after it). Before waiting on
// output tile-row m it serves every row through the end of the NEXT block: the rows of m's block are
// required (compute tilizes a whole block before its first output), the next block's are the
// prefetch that keeps reads overlapped with compute. Deadlock-free: the next block's first request
// needs compute to have released block(m) - 1, whose outputs this kernel has already drained.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (MaybeDeviceZoneScope, PERMANENT — perf tournaments rank stages from them; free when the
// profiler is off): per served split row writer_split_req_wait, writer_split_issue, writer_split_barrier;
// per output group writer_out_wait (starved on compute), writer_out_issue, writer_out_flush.
// Measurement-only ablations (compiled out unless defined; wrong results by construction):
// GQKV_ABLATE_READS drops the split reader's DRAM reads, GQKV_ABLATE_WRITES the q / k / v writes.
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
constexpr uint32_t cb_split_requests = 6;  // reader -> this kernel: one L1 row address per tile-row
constexpr uint32_t cb_split_done = 7;      // this kernel -> reader: one token per tile-row

constexpr uint32_t TILE_DIM = 32;

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t split_rows = get_compile_time_arg_val(6);         // 0: the reader reads every row
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);        // one token row of this core's channels
    constexpr uint32_t out_group_tiles = get_compile_time_arg_val(8);    // compute's push quantum
    constexpr bool stagger_row_walk = get_compile_time_arg_val(9) == 1;  // rotate each core's row order
    constexpr auto q_args = TensorAccessorArgs<10>();
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

    // store_tile: global channel tile cg of tile-row r -> its q / k / v page.
    auto store_tile = [&](uint32_t src, uint32_t r, uint32_t cg) {
        if (cg < q_tiles) {
            GQKV_WRITE(src, q_acc.get_noc_addr(r * q_tiles + cg), tile_bytes);
        } else if (cg < q_tiles + k_tiles) {
            GQKV_WRITE(src, k_acc.get_noc_addr(r * k_tiles + (cg - q_tiles)), tile_bytes);
        } else {
            GQKV_WRITE(src, v_acc.get_noc_addr(r * v_tiles + (cg - q_tiles - k_tiles)), tile_bytes);
        }
    };

    // ---- split reader: serve tile-rows [0, rows_served) of this core's range ----
    [[maybe_unused]] uint32_t rows_served = 0;
    [[maybe_unused]] auto serve_split_rows_through = [&](uint32_t end) {
        if constexpr (split_rows > 0) {
            const auto input_acc = TensorAccessor(input_args, input_addr);
            const uint32_t input_byte_offset =
                (input_col_offset_tiles + ch_group_tile_start) * (stick_bytes / block_ch_tiles);
            for (; rows_served < end; ++rows_served) {
                {
                    MaybeDeviceZoneScope("writer_split_req_wait");
                    cb_wait_front(cb_split_requests, 1);
                }
                invalidate_l1_cache();  // the address was stored by the reader RISC
                const uint32_t dst = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_split_requests));
                cb_pop_front(cb_split_requests, 1);
                const uint32_t token0 = (core_seq_start + rows_served) * TILE_DIM;
                {
                    MaybeDeviceZoneScope("writer_split_issue");
                    for (uint32_t j = 0; j < split_rows; ++j) {
                        const uint32_t i =
                            TILE_DIM - split_rows +
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
        }
    };

    for (uint32_t r = core_seq_start; r < core_seq_start + core_seq_tiles; ++r) {
        if constexpr (split_rows > 0) {
            const uint32_t m = r - core_seq_start;
            const uint32_t next_block_end = (m / block_rows + 2) * block_rows;
            serve_split_rows_through(next_block_end < core_seq_tiles ? next_block_end : core_seq_tiles);
        }
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
    }
    noc_async_write_barrier();
}
