// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// transfer_alignment writer (BRISC / NoC1).
//
// Drains each extracted span from the reader->writer CB and writes it back to its
// DRAM output page. The reader always lands the useful span at a residue-0 offset
// inside the CB slot, so this span->DRAM write is congruent (source residue 0 ==
// output page residue 0) and byte-identical for BOTH access-strategy variants — the
// strategy lives entirely in the reader (see ta_reader.cpp), so the writer never
// contributes to the measured delta. Runs on NoC1 so its writes overlap the reader's
// NoC0 reads.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_span = 16;

// Matches the reader's residue-0 landing so the writer reads exactly where the span
// was placed. `align` is a power of two.
static inline uint32_t landing_offset(uint32_t base, uint32_t align) {
    const uint32_t r = base & (align - 1);
    return r == 0 ? 0u : (align - r);
}

void kernel_main() {
    constexpr uint32_t span_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t align = get_compile_time_arg_val(1);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, out_page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            const uint32_t row = start_row + r;
            cb_wait_front(cb_span, 1);
            const uint32_t base = get_read_ptr(cb_span);
            const uint32_t off = landing_offset(base, align);  // same residue-0 landing the reader used
            noc_async_write(base + off, out_acc.get_noc_addr(row), span_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_span, 1);
        }
    }
}
