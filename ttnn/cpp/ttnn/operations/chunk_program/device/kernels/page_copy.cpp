// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/page_copy.PageCopy's kernel, verbatim (NC.7).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page
constexpr uint32_t B = get_compile_time_arg_val(1);     // pages a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);    // cores
constexpr uint32_t SLOT = (PAGE + 63) & ~63u;
constexpr auto s_args = TensorAccessorArgs<3>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);   // this core's index of NC
    const uint32_t s0 = get_arg_val<uint32_t>(3);  // first source page (per-run value)
    const uint32_t d0 = get_arg_val<uint32_t>(4);  // first destination page (per-run value)
    const uint32_t n = get_arg_val<uint32_t>(5);   // pages a block (per-run value)
    const uint32_t nb = get_arg_val<uint32_t>(6);  // blocks (per-run value)
    const uint32_t ss = get_arg_val<uint32_t>(7);  // source pages between blocks (per-run value)
    const uint32_t ds = get_arg_val<uint32_t>(8);  // destination pages between blocks (per-run value)
    const auto S = TensorAccessor(s_args, src, PAGE);
    const auto D = TensorAccessor(d_args, dst, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t total = n * nb;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    uint32_t g = lo;
    while (g < hi) {
        uint32_t nbat = 0;
        const uint32_t g0 = g;
        for (; nbat < B && g < hi; ++nbat, ++g) {
            const uint32_t blk = g / n, i = g - blk * n;
            noc_async_read(S.get_noc_addr(s0 + blk * ss + i), buf + nbat * SLOT, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < nbat; ++t) {
            const uint32_t gg = g0 + t, blk = gg / n, i = gg - blk * n;
            noc_async_write(buf + t * SLOT, D.get_noc_addr(d0 + blk * ds + i), PAGE);
        }
        noc_async_write_barrier();
    }
}
