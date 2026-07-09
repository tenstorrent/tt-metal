// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer for fused_gate: writes Wt output tiles per edge tile-row.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);

    constexpr auto out_args = TensorAccessorArgs<3>();

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);

    const auto out_gen = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t r = 0; r < num_rows; r++) {
        const uint32_t row = start_row + r;
        cb_wait_front(cb_out, Wt);
        uint32_t rd = get_read_ptr(cb_out);
        const uint32_t base = row * Wt;
        for (uint32_t t = 0; t < Wt; t++) {
            noc_async_write_tile(base + t, out_gen, rd);
            rd += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }
}
