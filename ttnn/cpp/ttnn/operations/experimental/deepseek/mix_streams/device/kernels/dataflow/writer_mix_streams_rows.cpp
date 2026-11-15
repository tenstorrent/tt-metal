// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// A several-row block's output: rows i*h .. i*h + h-1 of each mixed 32x32 tile are stream i's (h x 32) tile of
// that column.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t hc = get_compile_time_arg_val(1);
    constexpr uint32_t h = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    CircularBuffer out_cb(cb_out);
    constexpr uint32_t face_rows_bytes = h * 16 * sizeof(uint16_t);

    for (uint32_t c = start_tile; c < start_tile + num_tiles; ++c) {
        out_cb.wait_front(1);
        for (uint32_t i = 0; i < hc; ++i) {
            const uint32_t r0 = i * h;
            const uint32_t src = (r0 < 16 ? 0u : 1024u) + (r0 & 15u) * 32u;
            const uint32_t page = i * n_tiles + c;
            noc.async_write(out_cb, out, face_rows_bytes, {.offset_bytes = src}, {.page_id = page});
            noc.async_write(
                out_cb,
                out,
                face_rows_bytes,
                {.offset_bytes = src + 512},
                {.page_id = page, .offset_bytes = face_rows_bytes});
        }
        noc.async_write_barrier();
        out_cb.pop_front(1);
    }
}
