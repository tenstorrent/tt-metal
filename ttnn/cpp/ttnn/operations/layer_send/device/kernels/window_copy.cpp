// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/window_send.py _COPY_KERNEL (WindowSend.fused, the copy)
// (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a row (a page)
constexpr uint32_t B = get_compile_time_arg_val(1);     // rows read a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);    // copy cores
constexpr uint32_t NSEG = get_compile_time_arg_val(3);  // segments: the region's, then the ring pieces
constexpr auto o_args = TensorAccessorArgs<4>();
constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t region = get_arg_val<uint32_t>(1);
    const uint32_t ring = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);  // this core's index of NC
    constexpr uint32_t S0 = 4;                    // segment s: (src row, dst row, rows) at S0 + 3 s
    const auto O = TensorAccessor(o_args, rows, PAGE);
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto G = TensorAccessor(g_args, ring, PAGE);
    uint32_t total = 0;
    for (uint32_t s = 0; s < NSEG; ++s) {
        total += get_arg_val<uint32_t>(S0 + 3 * s + 2);
    }
    const uint32_t lo = total * k / NC;
    const uint32_t hi = total * (k + 1) / NC;
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;  // 64 B aligned, as the DRAM pages
    uint32_t g = 0;                                       // first row of segment s in the concatenated rows
    for (uint32_t s = 0; s < NSEG; ++s) {
        const uint32_t src0 = get_arg_val<uint32_t>(S0 + 3 * s);
        const uint32_t dst0 = get_arg_val<uint32_t>(S0 + 3 * s + 1);
        const uint32_t n = get_arg_val<uint32_t>(S0 + 3 * s + 2);
        const uint32_t b = lo > g ? lo : g;
        const uint32_t e = hi < g + n ? hi : g + n;
        for (uint32_t r = b; r < e; r += B) {
            const uint32_t nb = (e - r) < B ? (e - r) : B;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(O.get_noc_addr(src0 + r - g + i), buf + i * PAGE, PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t d = dst0 + r - g + i;
                noc_async_write(buf + i * PAGE, s == 0 ? R.get_noc_addr(d) : G.get_noc_addr(d), PAGE);
            }
            noc_async_write_barrier();
        }
        g += n;
    }
}
