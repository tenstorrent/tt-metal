// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Pops untilized 32x32 bf16 tiles and scatters
// their 32 rows as 64 B fragments into the row-major output (page = row,
// fragments are 64 B aligned). The owner of a row's last valid tile fills
// the row tail [valid(s)*32, T) with -inf so skipped tiles never read junk.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Hi = get_compile_time_arg_val(0);
constexpr uint32_t Sqt = get_compile_time_arg_val(1);
constexpr uint32_t Tt = get_compile_time_arg_val(2);
constexpr uint32_t Dt = get_compile_time_arg_val(3);
constexpr uint32_t chunk_t = get_compile_time_arg_val(4);
constexpr uint32_t page_bytes = get_compile_time_arg_val(5);  // T*2

constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_scratch = tt::CBIndex::c_17;  // writer-only -inf scratch
constexpr uint32_t frag_bytes = 64;  // 32 bf16 = one tile row

inline uint32_t valid(uint32_t s) {
    uint32_t v = chunk_t + s + 1;
    return v < Tt ? v : Tt;
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<6>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    constexpr uint32_t scratch_elems = 32 * 32;
    constexpr uint32_t scratch_bytes = scratch_elems * 2;
    const uint32_t scratch = get_write_ptr(cb_scratch);
    {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        for (uint32_t e = 0; e < scratch_elems; ++e) {
            p[e] = 0xFF80;  // bf16 -inf
        }
    }

    uint32_t s = 0, rowsum = 0;
    while (flat_start >= rowsum + valid(s)) {
        rowsum += valid(s);
        ++s;
    }
    uint32_t t = flat_start - rowsum;

    for (uint32_t i = 0; i < flat_count; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < 32; ++r) {
            uint64_t dst = out_acc.get_noc_addr(s * 32 + r, t * frag_bytes);
            noc_async_write(l1 + r * frag_bytes, dst, frag_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
        const bool last_in_row = (t == valid(s) - 1);
        if (last_in_row && valid(s) < Tt) {
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

        if (++t == valid(s)) {
            ++s;
            t = 0;
        }
    }
}
