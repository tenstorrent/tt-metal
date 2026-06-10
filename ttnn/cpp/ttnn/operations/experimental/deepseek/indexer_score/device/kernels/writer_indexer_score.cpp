// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Pops untilized 32x32 bf16 tiles and scatters
// their 32 rows as 64 B fragments into the row-major output (page = row,
// fragments are 64 B aligned). The owner of a row's last valid tile fills
// the row tail [valid(s)*32, T) with -inf so skipped tiles never read junk.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"

constexpr uint32_t page_bytes = get_compile_time_arg_val(8);  // T*2

constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_scratch = tt::CBIndex::c_17;  // writer-only -inf scratch tile

constexpr uint32_t frag_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);  // one bf16 tile row
constexpr uint32_t scratch_bytes = tt::constants::TILE_HW * sizeof(uint16_t);

/** Stamp the -inf scratch tile (plain L1 stores) and return its address. */
inline uint32_t fill_inf_scratch() {
    fill_neginf_tile<scratch_bytes>(cb_scratch, /*tile_id=*/0);
    return CircularBuffer(cb_scratch).get_write_ptr();
}

/** Scatter one untilized tile's rows into output rows of q-tile-row s, column tile t. */
template <typename OutAcc>
inline void write_tile(Noc noc, const OutAcc& out_acc, uint32_t s, uint32_t t) {
    CircularBuffer cb(cb_out);
    cb.wait_front(1);
    uint32_t src = cb.get_read_ptr();
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        noc.async_write(
            CoreLocalMem<uint32_t>(src),
            out_acc,
            frag_bytes,
            {},
            {.page_id = s * tt::constants::TILE_HEIGHT + r, .offset_bytes = t * frag_bytes});
        src += frag_bytes;
    }
    noc.async_write_barrier();
    cb.pop_front(1);
}

/** Fill row tail [valid(s)*32, T) of q-tile-row s with -inf from the L1 scratch tile. */
template <typename OutAcc>
inline void fill_row_tail(Noc noc, const OutAcc& out_acc, uint32_t scratch, uint32_t s) {
    const uint32_t tail_bytes_total = (Tt - valid(s)) * frag_bytes;
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        uint32_t off = valid(s) * frag_bytes;
        uint32_t left = tail_bytes_total;
        while (left > 0) {
            const uint32_t n = left < scratch_bytes ? left : scratch_bytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(scratch),
                out_acc,
                n,
                {},
                {.page_id = s * tt::constants::TILE_HEIGHT + r, .offset_bytes = off});
            off += n;
            left -= n;
        }
    }
    noc.async_write_barrier();
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<9>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    Noc noc;

    const uint32_t scratch = fill_inf_scratch();

    WorkUnitSpan span;
    span.start(flat_start);

    for (uint32_t i = 0; i < flat_count; ++i) {
        // compute emits unit tiles in (r, c) row-major order
        for (uint32_t r = 0; r < QC; ++r) {
            for (uint32_t c = 0; c < span.kw(); ++c) {
                write_tile(noc, out_acc, span.s0() + r, span.c0() + c);
            }
        }
        if (span.last_in_group()) {
            for (uint32_t r = 0; r < QC; ++r) {
                const uint32_t s = span.s0() + r;
                if (valid(s) < Tt) {
                    fill_row_tail(noc, out_acc, scratch, s);
                }
            }
        }
        span.advance();
    }
}
