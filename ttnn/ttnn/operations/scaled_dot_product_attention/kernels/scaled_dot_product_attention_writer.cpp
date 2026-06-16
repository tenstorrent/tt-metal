// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention writer kernel.
//
// Drains cb_out (one Q-chunk of d_t head tiles per work-unit, in unit order)
// to the output tensor O (B, H_q, S_q, D).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H_q = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(1);
    constexpr uint32_t d_t = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t qc = u % Sq_t;
        const uint32_t tmp = u / Sq_t;
        const uint32_t h = tmp % H_q;
        const uint32_t b = tmp / H_q;

        const uint32_t out_base = ((b * H_q + h) * Sq_t + qc) * d_t;

        cb_wait_front(cb_out, d_t);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t dd = 0; dd < d_t; ++dd) {
            noc_async_write_page(out_base + dd, out_acc, rd);
            rd += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, d_t);
    }
}
