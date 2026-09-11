// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

// Unpack each 32x32 mixed tile back to ROW_MAJOR: the first ``hc`` rows, 32 columns,
// are written into the matching RM pages (one page per stream row).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t hc = get_compile_time_arg_val(1);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t half_row_bytes = 16 * sizeof(uint16_t);
    constexpr uint32_t row_slice_bytes = 32 * sizeof(uint16_t);

    for (uint32_t page = start_tile; page < start_tile + num_tiles; ++page) {
        out_cb.wait_front(one_tile);
        const uint32_t t = page / n_tiles;
        const uint32_t n_idx = page % n_tiles;
        const uint32_t col_bytes = n_idx * row_slice_bytes;
        for (uint32_t r = 0; r < hc; ++r) {
            const uint32_t left_off = tile_face_index(r, 0) * sizeof(uint16_t);
            const uint32_t right_off = tile_face_index(r, 16) * sizeof(uint16_t);
            noc.async_write(
                out_cb,
                out,
                half_row_bytes,
                {.offset_bytes = left_off},
                {.page_id = t * hc + r, .offset_bytes = col_bytes});
            noc.async_write(
                out_cb,
                out,
                half_row_bytes,
                {.offset_bytes = right_off},
                {.page_id = t * hc + r, .offset_bytes = col_bytes + half_row_bytes});
        }
        noc.async_write_barrier();
        out_cb.pop_front(one_tile);
    }
}
