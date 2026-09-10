// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// reader_state_reuse bench — shared writer (BRISC), byte-identical across every
// reader variant. `read_sticks_for_tilize` (the baseline) reserves/pushes
// `block_width_tiles` (`bw`) pages ONE TILE-ROW (32 sticks) AT A TIME even
// when a block spans several tile-rows (`tilize_helpers_dataflow.inl:112-134`,
// `total_blocks = div_up(total_num_rows, tile_h)`) -- so this writer (and
// every non-helper reader variant, to stay a fair comparison at the SAME CB
// granularity) chunks the same way: `rows / 32` reserve/write/pop cycles, 32
// sticks each. Drains one chunk from cb_in and stores it back out as 32
// ordinary interleaved pages. Fixed cost, held constant across every reader
// variant (perf-lab concept isolation) -- any measured delta between reader
// variants is attributable to the reader alone.

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kTileH = 32;
}

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t bw = get_compile_time_arg_val(2);  // tile-pages per chunk (block_width_tiles)
    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr uint32_t num_chunks = rows / kTileH;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);

    const auto out_acc = TensorAccessor(out_args, dst_addr, row_bytes);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        cb_wait_front(cb_in, bw);
        const uint32_t l1_read_addr = get_read_ptr(cb_in);
        const uint32_t row_base = start_page + chunk * kTileH;
        for (uint32_t r = 0; r < kTileH; ++r) {
            noc_async_write(l1_read_addr + r * row_bytes, out_acc.get_noc_addr(row_base + r), row_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_in, bw);
    }
}
