// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Pops untilized bf16 strips and scatters each strip's
// 32 rows into the row-major output (page = row, one contiguous run per row).
// The owner of a group's last unit fills row tails [valid_k_tiles*32, T)
// with -inf; compute already covers [0, valid_k_tiles) for every row.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk

constexpr uint32_t page_bytes = get_compile_time_arg_val(num_common_ct_args);  // row-major page = T*2 bytes

constexpr uint32_t frag_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);  // one bf16 tile row
constexpr uint32_t scratch_bytes = tt::constants::TILE_HW * sizeof(uint16_t);

/** Stamp the -inf scratch tile (plain L1 stores) and return its L1 address for reuse as a fill source. */
inline uint32_t fill_inf_scratch_and_get_addr() {
    fill_neginf_tile<scratch_bytes>(cb_scratch, /*tile_id=*/0);
    return CircularBuffer(cb_scratch).get_write_ptr();
}

/** Scatter one strip into the 32 output rows of q-tile-row q_row at column tile k_tile_start. Strip is
 *  always KC tiles wide; each row written as ONE contiguous run (1 async_write/row, not KC fragments).
 *  `valid_w` = KC tiles inside T (< KC on a partial last unit; KC need not divide Tt). CB always pops
 *  full KC; only the in-bounds prefix is written. */
template <typename OutAcc>
inline void write_strip(Noc noc, const OutAcc& out_acc, uint32_t q_row, uint32_t k_tile_start, uint32_t valid_w) {
    CircularBuffer cb(cb_out_strip);
    cb.wait_front(k_tiles_per_unit);
    uint32_t src = cb.get_read_ptr();
    const uint32_t row_pitch = k_tiles_per_unit * frag_bytes;  // strip row stride (full KC)
    const uint32_t write_bytes = valid_w * frag_bytes;         // only the in-bounds columns
    for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
        noc.async_write(
            CoreLocalMem<uint32_t>(src),
            out_acc,
            write_bytes,
            {},
            {.page_id = q_row * tt::constants::TILE_HEIGHT + rr, .offset_bytes = k_tile_start * frag_bytes});
        src += row_pitch;
    }
    noc.async_write_barrier();
    cb.pop_front(k_tiles_per_unit);
}

/** Fill the group's row tails [tail_start_tiles*32, T) with -inf from the L1 scratch tile.
 *  Tail starts at the group's valid width (same column for every row), NOT each row's own valid
 *  length -- this keeps the fill disjoint from compute's output, so cores sharing a group never
 *  double-write the same bytes. */
template <typename OutAcc>
inline void fill_group_tails(
    Noc noc, const OutAcc& out_acc, uint32_t scratch, uint32_t q_tile_start, uint32_t tail_start_tiles) {
    if (tail_start_tiles >= k_len_tiles) {
        return;  // group is fully causal-valid, no tail
    }
    const uint32_t tail_bytes_total = (k_len_tiles - tail_start_tiles) * frag_bytes;
    const uint32_t base_off = tail_start_tiles * frag_bytes;
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
            uint32_t off = base_off;
            uint32_t left = tail_bytes_total;
            while (left > 0) {
                // chunk == scratch tile size: the -inf source is exactly one tile, so each write
                // copies at most a tile-worth of bytes before reusing the same source.
                const uint32_t n = left < scratch_bytes ? left : scratch_bytes;
                noc.async_write(
                    CoreLocalMem<uint32_t>(scratch),
                    out_acc,
                    n,
                    {},
                    {.page_id = (q_tile_start + r) * tt::constants::TILE_HEIGHT + rr, .offset_bytes = off});
                off += n;
                left -= n;
            }
        }
    }
    noc.async_write_barrier();
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<num_common_ct_args + 1>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    Noc noc;

    const uint32_t scratch = fill_inf_scratch_and_get_addr();

    WorkUnitSpan span;
    span.start(flat_start);

    for (uint32_t i = 0; i < flat_count; ++i) {
        const uint32_t k_tile0 = span.k_tile_start();
        const uint32_t valid_w = span.k_tiles();  // == KC for interior units, < KC for a partial last unit
        // One KC-wide strip per row; masked suffix already stamped by compute. Write only valid_w columns.
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            write_strip(noc, out_acc, span.q_tile_start() + r, k_tile0, valid_w);
        }
        if (span.last_in_group()) {
            fill_group_tails(noc, out_acc, scratch, span.q_tile_start(), span.valid_k_tiles());
        }
        span.advance();
    }
}
