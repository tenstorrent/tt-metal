// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_pad extension_band writer (BRISC / NoC1): writes c into tile rows [a, b) of the last tile
// row of every slice, in place. The output is a view of the input's buffer, so nothing is read
// from DRAM and no other tile row is touched.
//
// write_band_block uses raw noc_async_write: no kernel_lib helper writes a constant byte run at
// an in-page offset of an interleaved page for c != 0 (Noc::async_write_zeros covers zeros only).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_pad_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_fill_page_writer = get_compile_time_arg_val(0);
    constexpr uint32_t es = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr bool fill_is_zero = get_compile_time_arg_val(3) != 0;
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t ht_in = get_common_arg_val<uint32_t>(2);
    const uint32_t wt_in = get_common_arg_val<uint32_t>(3);
    const uint32_t num_runs = get_common_arg_val<uint32_t>(4);
    uint32_t run_off[3], run_len[3];
    for (uint32_t j = 0; j < 3; ++j) {
        run_off[j] = get_common_arg_val<uint32_t>(5 + 2 * j);
        run_len[j] = get_common_arg_val<uint32_t>(6 + 2 * j);
    }

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t units_this_core = get_arg_val<uint32_t>(1);
    if (units_this_core == 0) {
        return;
    }

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    // Runs source the fill page at the offset they target, so only the span of the runs is built.
    const uint32_t span_start = run_off[0];
    const uint32_t span_end = run_off[num_runs - 1] + run_len[num_runs - 1];
    {
        DeviceZoneScopedN("G6PAD-FILL-PAGE");
        g6_pad::build_fill_page<cb_fill_page_writer, es, fill_is_zero>(span_start, span_end - span_start, c_bits);
    }
    const uint32_t fill_page = get_write_ptr(cb_fill_page_writer);

    {
        DeviceZoneScopedN("G6PAD-BAND");
        uint32_t slice = unit_start / wt_in;
        uint32_t wt = unit_start - slice * wt_in;
        uint32_t page_id = (slice * ht_in + ht_in - 1) * wt_in + wt;
        const uint32_t slice_stride = ht_in * wt_in;
        for (uint32_t u = 0; u < units_this_core; ++u) {
            for (uint32_t j = 0; j < num_runs; ++j) {
                noc_async_write(fill_page + run_off[j], out.get_noc_addr(page_id, run_off[j]), run_len[j]);
            }
            ++page_id;
            if (++wt == wt_in) {
                wt = 0;
                page_id += slice_stride - wt_in;
            }
        }
    }
    {
        DeviceZoneScopedN("G6PAD-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
