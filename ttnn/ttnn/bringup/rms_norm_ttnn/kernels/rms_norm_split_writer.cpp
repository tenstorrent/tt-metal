// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.rms_norm, the split program's writer (CHANGELOG 6). Unit u = (row tile r = u / B, column block
// b = u % B). MODE_STATS_OUT: one fp32 partial tile per unit at page r B + b of the [.., R, 32 B] statistics;
// otherwise the unit's WB output tiles at pages r WT_OUT + b WB + j.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t MODE_STATS_OUT = 1;

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t WT_OUT = get_compile_time_arg_val(1);
    constexpr uint32_t WB = get_compile_time_arg_val(2);
    constexpr uint32_t BLK = get_compile_time_arg_val(3);
    constexpr uint32_t B = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_end = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cbo(cb_out);

    for (uint32_t u = unit_start; u < unit_end; ++u) {
        const uint32_t r = u / B;
        const uint32_t b = u % B;
        if constexpr (MODE == MODE_STATS_OUT) {
            cbo.wait_front(1);
            noc.async_write(cbo, out_acc, tile_bytes, {.offset_bytes = 0}, {.page_id = r * B + b});
            noc.async_writes_flushed();
            cbo.pop_front(1);
            continue;
        }
        const uint32_t first = r * WT_OUT + b * WB;
        for (uint32_t col = 0; col < WB; col += BLK) {
            cbo.wait_front(BLK);
            uint32_t off = 0;
            for (uint32_t i = 0; i < BLK; ++i) {
                noc.async_write(cbo, out_acc, tile_bytes, {.offset_bytes = off}, {.page_id = first + col + i});
                off += tile_bytes;
            }
            noc.async_writes_flushed();
            cbo.pop_front(BLK);
        }
    }
    noc.async_write_barrier();
}
