// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Pops untilized 32x32 bf16 tiles and scatters
// their 32 rows as 64 B fragments into the row-major output (page = row,
// fragments are 64 B aligned). The owner of a row's last valid tile fills
// the row tail [valid(s)*32, T) with -inf so skipped tiles never read junk.

#include "api/dataflow/dataflow_api.h"

#include "indexer_score_common.hpp"

constexpr uint32_t page_bytes = get_compile_time_arg_val(5);  // T*2

constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_scratch = tt::CBIndex::c_17;  // writer-only -inf scratch
constexpr uint32_t frag_bytes = 64;  // 32 bf16 = one tile row

constexpr uint32_t scratch_elems = 32 * 32;
constexpr uint32_t scratch_bytes = scratch_elems * 2;

inline uint32_t fill_inf_scratch() {
    const uint32_t scratch = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
    for (uint32_t e = 0; e < scratch_elems; ++e) {
        p[e] = 0xFF80;  // bf16 -inf
    }
    return scratch;
}

// scatter one untilized tile's 32 rows into output rows of q-tile-row s
template <typename OutAcc>
inline void write_tile(const OutAcc& out_acc, uint32_t s, uint32_t t) {
    cb_wait_front(cb_out, 1);
    uint32_t l1 = get_read_ptr(cb_out);
    for (uint32_t r = 0; r < 32; ++r) {
        uint64_t dst = out_acc.get_noc_addr(s * 32 + r, t * frag_bytes);
        noc_async_write(l1 + r * frag_bytes, dst, frag_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}

// fill row tail [valid(s)*32, T) with -inf from the L1 scratch tile
template <typename OutAcc>
inline void fill_row_tail(const OutAcc& out_acc, uint32_t scratch, uint32_t s) {
    const uint32_t tail_bytes_total = (Tt - valid(s)) * frag_bytes;
    for (uint32_t r = 0; r < 32; ++r) {
        uint32_t off = valid(s) * frag_bytes;
        uint32_t left = tail_bytes_total;
        while (left > 0) {
            uint32_t n = left < scratch_bytes ? left : scratch_bytes;
            noc_async_write(scratch, out_acc.get_noc_addr(s * 32 + r, off), n);
            off += n;
            left -= n;
        }
    }
    noc_async_write_barrier();
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<6>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    const uint32_t scratch = fill_inf_scratch();

    ValidTileSpan span;
    span.start(flat_start);

    for (uint32_t i = 0; i < flat_count; ++i) {
        write_tile(out_acc, span.s, span.t);
        if (span.last_in_row() && valid(span.s) < Tt) {
            fill_row_tail(out_acc, scratch, span.s);
        }
        span.advance();
    }
}
