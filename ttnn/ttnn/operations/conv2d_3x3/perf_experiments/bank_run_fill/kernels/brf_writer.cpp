// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_run_fill bench, writer (BRISC, NoC1): consumes the slab (verify: writes each band's pixels to output page
// (core * max_blocks + m) * 3 + kh; timing: just pops). Reader variants: one whole slab per block, band kh at
// kh * band_cap pixels. TRISC variants: 256-pixel groups, band by band, in chunk order.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "brf_geom.hpp"

constexpr uint32_t CB_SLAB = 0;

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    using G = brf::Geom<
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7)>;
    constexpr uint32_t pb = get_compile_time_arg_val(8);
    constexpr uint32_t band_cap = get_compile_time_arg_val(9);
    constexpr bool verify = get_compile_time_arg_val(10) != 0;
    constexpr auto out_args = TensorAccessorArgs<11>();
    constexpr bool trisc = variant == 4 || variant == 7;
    constexpr uint32_t cit = pb / 64;
    constexpr uint32_t slab_pages = 3 * band_cap * pb / 2048;
    constexpr uint32_t page_bytes = band_cap * pb;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t actual = get_arg_val<uint32_t>(2);
    const uint32_t out_page_base = get_arg_val<uint32_t>(3);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);
    const bool write_here = verify && out_page_base != 0xFFFFFFFFu;  // verify a subset of cores

    const uint32_t num_blocks = (actual + G::bm - 1) / G::bm;
    for (uint32_t m = 0; m < num_blocks; ++m) {
        const brf::Bands b = G::bands(m_tile_start, actual, m);
        if constexpr (!trisc) {
            cb_wait_front(CB_SLAB, slab_pages);
            if (write_here) {
                const uint32_t slab = get_read_ptr(CB_SLAB);
                for (uint32_t kh = 0; kh < 3; ++kh) {
                    const uint32_t n = static_cast<uint32_t>(b.hi[kh] - b.lo[kh]);
                    if (n > 0) {
                        noc_async_write(
                            slab + kh * band_cap * pb, out.get_noc_addr(out_page_base + m * 3 + kh), n * pb);
                    }
                }
                noc_async_write_barrier();
            }
            cb_pop_front(CB_SLAB, slab_pages);
        } else {
            for (uint32_t kh = 0; kh < 3; ++kh) {
                const uint32_t n = static_cast<uint32_t>(b.hi[kh] - b.lo[kh]);
                for (uint32_t px = 0; px < n; px += 256) {
                    cb_wait_front(CB_SLAB, 8 * cit);
                    if (write_here) {
                        const uint32_t cnt = n - px < 256 ? n - px : 256;
                        noc_async_write(
                            get_read_ptr(CB_SLAB), out.get_noc_addr(out_page_base + m * 3 + kh, px * pb), cnt * pb);
                        noc_async_write_barrier();
                    }
                    cb_pop_front(CB_SLAB, 8 * cit);
                }
            }
        }
    }
}
