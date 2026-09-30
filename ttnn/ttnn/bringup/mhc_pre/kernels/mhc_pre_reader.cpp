// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre reader (NCRISC; NoC0, or NoC1 for the flipped top groups — READER_NOC_FLIP_FRACTION): the X stream, and
// (W column all-gather, W_SHARE_ON_READER) this core's W column share, issued ahead of the X burst on the same NoC
// (transaction id 15) so it rides the row's uncongested NoC; it lands in the writer-owned cb_weight at the writer's
// slots and is handed over by token (cb_w_share_landed) before the first X chunk is waited for. The writer does
// the split / multicast / publish and reads the bias (and, R1, the whole W slice).
//
// load_resident_constants (once): reduce scalers (SUM / REDUCE_ROW, 1.0; fp32 X also MAX / REDUCE_SCALAR).
// load_x_block (per block): block_token_tiles x core_k_tiles X tiles of this rank's stream-column slice,
//   L1 slot (t, c, i) = t*core_k_tiles + c*n + i. Read in x_stream_chunks slot-ordered chunks with at most
//   x_stream_inflight chunks outstanding: chunk j's reads carry transaction id j + 1, and chunk j is pushed as
//   soon as its own reads landed (the compute's projection + sum x^2 wait per K chunk, so they run under the
//   rest of the burst).
//   The CB is always pushed by the NOMINAL block size (block_token_tiles * core_k_tiles_max; the last chunk
//   carries the padding) so the FIFO never wraps inside a block, whatever this rank's core_k_tiles or the
//   ragged last block's extent.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via the KERNEL_PERF_ZONES define, see perf_instrumentation.hpp):
//   r_w_issue / r_w_land     W column share issue / its barrier (trid 15)
//   r_x_reserve              cb_x_resident back-pressure (the compute has not freed block b-2)
//   r_x_issue / r_x_barrier  per X chunk: address generation + issue / waiting for its reads to land
// Ablation (perf tournaments only): MHC_ABLATE_XREAD skips the X DRAM reads (pushes unchanged).

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t n_streams = get_compile_time_arg_val(2);
    constexpr uint32_t block_token_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t core_k_tiles_max = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(6);
    constexpr bool needs_max_scaler = get_compile_time_arg_val(7) != 0;  // fp32 X grid split
    constexpr uint32_t x_stream_chunks = get_compile_time_arg_val(8);    // 1 = one publish per block
    static_assert(x_stream_chunks >= 1 && x_stream_chunks <= 14, "one NoC transaction id per chunk (1..14)");
    constexpr uint32_t w_share_trid = 15;  // the W column share's reads (issued ahead of the X burst)
    constexpr uint32_t x_stream_inflight = get_compile_time_arg_val(9);  // chunks outstanding (>= chunks: all)
    static_assert(x_stream_inflight >= 1, "at least one chunk in flight");
    constexpr uint32_t cb_weight = get_compile_time_arg_val(10);          // W share landing (writer-owned CB)
    constexpr uint32_t cb_w_share_landed = get_compile_time_arg_val(11);  // token: reader -> writer
    constexpr bool w_share_before_x = get_compile_time_arg_val(12) != 0;  // land the share before any X read
    constexpr auto x_args = TensorAccessorArgs<13>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t t_start = get_arg_val<uint32_t>(1);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(2);
    const uint32_t c_start = get_arg_val<uint32_t>(3);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(4);
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);
    const uint32_t w_addr = get_arg_val<uint32_t>(6);
    const bool w_share_on_reader = get_arg_val<uint32_t>(7) != 0;  // W_ROLE_SPREAD: this core reads its share
    const uint32_t own_p0 = get_arg_val<uint32_t>(8);              // share [own_p0, own_p1) of the slice
    const uint32_t own_p1 = get_arg_val<uint32_t>(9);

    constexpr uint32_t tensor_k_tiles = n_streams * tensor_c_tiles;
    constexpr uint32_t x_block_pages = block_token_tiles * core_k_tiles_max;  // nominal push per block
    const uint32_t core_k_tiles = n_streams * core_c_tiles;
    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);

    bool w_share_pending = w_share_on_reader;
    auto land_w_share = [&]() {  // before the first X chunk is waited for (the share was issued ahead of it)
        if (w_share_pending) {
            MaybeDeviceZoneScope("r_w_land");
            noc_async_read_barrier_with_trid(w_share_trid);
            cb_reserve_back(cb_w_share_landed, 1);
            cb_push_back(cb_w_share_landed, 1);
            w_share_pending = false;
        }
    };
    auto load_x_block = [&](uint32_t block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;
        const uint32_t real_pages = extent * core_k_tiles;
        const uint32_t chunk_pages = (real_pages + x_stream_chunks - 1) / x_stream_chunks;
        const uint32_t num_chunks = chunk_pages == 0 ? 0 : (real_pages + chunk_pages - 1) / chunk_pages;
        {
            MaybeDeviceZoneScope("r_x_reserve");
            cb_reserve_back(cb_x_resident, x_block_pages);
        }
        const uint32_t base = get_write_ptr(cb_x_resident);
        // Issue cursor over L1 slot p = t*core_k_tiles + c*n + i (DRAM page row(t) + i*Ct + c).
        uint32_t p = 0, t = 0, c = 0, i = 0;
        uint32_t row_page = (t_start + row0) * tensor_k_tiles + c_start;
        auto issue_chunk = [&](uint32_t j) {
            MaybeDeviceZoneScope("r_x_issue");
            const uint32_t end = (j + 1) * chunk_pages < real_pages ? (j + 1) * chunk_pages : real_pages;
            noc_async_read_set_trid(1 + j);  // sticky for this chunk's reads
            for (; p < end; ++p) {
#ifndef MHC_ABLATE_XREAD
                noc_async_read_page(row_page + i * tensor_c_tiles + c, x_acc, base + p * x_tile_bytes);
#endif
                if (++i == n_streams) {
                    i = 0;
                    if (++c == core_c_tiles) {
                        c = 0;
                        ++t;
                        row_page += tensor_k_tiles;
                    }
                }
            }
            noc_async_read_set_trid(0);
        };
        uint32_t pushed = 0;
        auto publish_chunk = [&](uint32_t j) {
            land_w_share();
            {
                MaybeDeviceZoneScope("r_x_barrier");
                noc_async_read_barrier_with_trid(1 + j);
            }
            // the last chunk carries the nominal-size padding
            const uint32_t end = (j + 1) * chunk_pages < real_pages ? (j + 1) * chunk_pages : x_block_pages;
            cb_push_back(cb_x_resident, end - pushed);
            pushed = end;
        };
        // At most x_stream_inflight chunks outstanding: the DRAM banks then serve this core's chunks roughly in
        // order (all issued at once, every chunk lands at ~the end of the burst), so the compute's per-K-tile
        // projection runs under the rest of the stream.
        for (uint32_t j = 0; j < num_chunks; ++j) {
            if (j >= x_stream_inflight) {
                publish_chunk(j - x_stream_inflight);
            }
            issue_chunk(j);
        }
        for (uint32_t j = num_chunks > x_stream_inflight ? num_chunks - x_stream_inflight : 0; j < num_chunks; ++j) {
            publish_chunk(j);
        }
        if (pushed < x_block_pages) {  // empty block (never issued): keep the nominal push count
            cb_push_back(cb_x_resident, x_block_pages - pushed);
        }
    };

    // Scalers first (a local zero fill + a few stores, well under a microsecond), then the X stream.
    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    if constexpr (needs_max_scaler) {
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_SCALAR>(1.0f);
    }
    // W column share (W_ROLE_SPREAD): issued ahead of the X burst on this reader's NoC; it lands in the writer's
    // cb_weight at the same slots the writer would use (cb_weight's base: the writer reserves the whole slice
    // once and never wraps), and the writer is told by token once it landed (it splits / multicasts / publishes).
    if (w_share_on_reader) {
        MaybeDeviceZoneScope("r_w_issue");
        const uint32_t w_tile_bytes = get_tile_size(cb_weight);
        const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
        const uint32_t w_base = get_write_ptr(cb_weight);
        noc_async_read_set_trid(w_share_trid);
        for (uint32_t p = own_p0; p < own_p1; ++p) {
            const uint32_t c = p / n_streams;
            const uint32_t i = p - c * n_streams;
            noc_async_read_page(i * tensor_c_tiles + c_start + c, w_acc, w_base + p * w_tile_bytes);
        }
        noc_async_read_set_trid(0);
        if constexpr (w_share_before_x) {
            land_w_share();
        }
    }
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        load_x_block(block_idx);
    }
    land_w_share();  // (no X chunk was ever waited for)
}
