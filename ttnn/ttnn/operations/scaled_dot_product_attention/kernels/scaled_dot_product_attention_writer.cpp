// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flash-Attention SDPA writer (BRISC).
//
// Per work unit (b, h, q): drain the normalized output tile-row cb_out (Dt bf16
// tiles, TileRowMajor [1, Dt]) and write to O[b, h, q*32:(q+1)*32, :] —
// tile (b,h,q,nd) at page ((b*H + h)*Sq_t + q)*Dt + nd.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);

    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    const uint32_t page_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    const auto o_acc = TensorAccessor(out_args, o_addr, page_bytes);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t unit = start_unit + u;
        const uint32_t b = unit / (H * Sq_t);
        const uint32_t h = (unit / Sq_t) % H;
        const uint32_t q = unit % Sq_t;

        const uint32_t o_row_base = ((b * H + h) * Sq_t + q) * Dt;  // tile (b,h,q,0)

        out_cb.wait_front(Dt);
        for (uint32_t nd = 0; nd < Dt; ++nd) {
            noc.async_write(out_cb, o_acc, page_bytes, {.offset_bytes = nd * page_bytes}, {.page_id = o_row_base + nd});
        }
        noc.async_write_barrier();
        out_cb.pop_front(Dt);
    }
}
