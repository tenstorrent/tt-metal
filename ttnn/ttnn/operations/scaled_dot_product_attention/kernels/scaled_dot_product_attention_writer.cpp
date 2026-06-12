// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention writer (BRISC).
//
// For each (b, h, q_block) work unit owned by this core, drains the normalized
// output block O_i (D_t tiles, q_chunk_t == 1) from cb_out to the output DRAM
// buffer. Output tile_id matches Q's layout:
//     tile_id = ((b*H + h)*S_q_t + qi)*D_t + d

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t D_t = get_compile_time_arg_val(0);
    constexpr uint32_t S_q_t = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto dst_acc = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t idx = 0; idx < num_units; ++idx) {
        const uint32_t u = start_unit + idx;
        const uint32_t qi = u % S_q_t;
        const uint32_t bh = u / S_q_t;
        const uint32_t h = bh % H;
        const uint32_t b = bh / H;

        const uint32_t base = ((b * H + h) * S_q_t + qi) * D_t;
        cb_wait_front(cb_out, D_t);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t d = 0; d < D_t; ++d) {
            noc_async_write_tile(base + d, dst_acc, l1 + d * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, D_t);
    }
}
