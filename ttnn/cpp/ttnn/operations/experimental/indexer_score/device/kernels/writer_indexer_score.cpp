// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Pops untilized bf16 strips and scatters each strip's
// 32 rows into the row-major output (page = row, one contiguous run per row). Under the dense schedule
// compute covers every column of every row (future keys already stamped to -inf), so the writer just
// scatters each unit's strip -- there is no separate row-tail fill.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>

#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk

constexpr uint32_t page_bytes = get_compile_time_arg_val(num_common_ct_args);  // row-major page = T*2 bytes

constexpr uint32_t frag_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);  // one bf16 tile row

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

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<num_common_ct_args + 1>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    Noc noc;

    WorkUnitSpan span;
    span.start(flat_start);

    for (uint32_t i = 0; i < flat_count; ++i) {
        const uint32_t k_tile0 = span.k_tile_start();
        const uint32_t valid_w = span.k_tiles();  // == KC for interior units, < KC for a partial last unit
        // One KC-wide strip per row; masked suffix already stamped by compute. Write only valid_w columns.
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            write_strip(noc, out_acc, span.q_tile_start() + r, k_tile0, valid_w);
        }
        span.advance();
    }
}
