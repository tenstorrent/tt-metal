// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NC.9: several page copies of one page size in one program (page_copy.cpp's loop a segment): segment s copies src_s's
// pages [s0_s, s0_s + n_s) to dst_s's pages [d0_s, d0_s + n_s). The segments' pages, one after the other, are split
// evenly over the NC cores (as page_copy.cpp splits one copy's). Every source shares one accessor config, every
// destination another (checked by the op).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page
constexpr uint32_t B = get_compile_time_arg_val(1);     // pages a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);    // cores
constexpr uint32_t NSEG = get_compile_time_arg_val(3);  // segments
constexpr uint32_t SLOT = (PAGE + 63) & ~63u;
constexpr auto s_args = TensorAccessorArgs<4>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t k = get_arg_val<uint32_t>(0);  // this core's index of NC
    // per segment (runtime args 1 + 5 s ..): src, dst, s0, d0, n (addresses and per-run values)
    uint32_t total = 0;
    for (uint32_t s = 0; s < NSEG; ++s) {
        total += get_arg_val<uint32_t>(1 + (5 * s) + 4);
    }
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    uint32_t base = 0;  // the segment's first page in the concatenation
    for (uint32_t s = 0; s < NSEG && base < hi; ++s) {
        const uint32_t a = 1 + (5 * s);
        const uint32_t n = get_arg_val<uint32_t>(a + 4);
        const uint32_t e = base + n;
        const uint32_t g_lo = lo > base ? lo : base;
        const uint32_t g_hi = hi < e ? hi : e;
        if (g_lo < g_hi) {
            const auto S = TensorAccessor(s_args, get_arg_val<uint32_t>(a), PAGE);
            const auto D = TensorAccessor(d_args, get_arg_val<uint32_t>(a + 1), PAGE);
            const uint32_t s0 = get_arg_val<uint32_t>(a + 2) - base;
            const uint32_t d0 = get_arg_val<uint32_t>(a + 3) - base;
            uint32_t g = g_lo;
            while (g < g_hi) {
                uint32_t nbat = 0;
                const uint32_t g0 = g;
                for (; nbat < B && g < g_hi; ++nbat, ++g) {
                    noc_async_read(S.get_noc_addr(s0 + g), buf + nbat * SLOT, PAGE);
                }
                noc_async_read_barrier();
                for (uint32_t t = 0; t < nbat; ++t) {
                    noc_async_write(buf + t * SLOT, D.get_noc_addr(d0 + g0 + t), PAGE);
                }
                noc_async_write_barrier();
            }
        }
        base = e;
    }
}
