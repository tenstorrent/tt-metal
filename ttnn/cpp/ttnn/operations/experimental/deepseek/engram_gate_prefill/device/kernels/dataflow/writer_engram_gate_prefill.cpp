// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// Output store. This core owns a whole unit and the full D, so there is nothing to gather: it writes each chunk of
// its slice straight into the unit's tile row.
//
//   unit = block * hc + h  ->  streams tile row (h * blocks + block), columns [chunk*d_tiles, ...)
void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t hc = get_compile_time_arg_val(3);
    constexpr uint32_t blocks = get_compile_time_arg_val(4);

    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);

    const auto out = TensorAccessor(out_args, out_addr);

    constexpr uint32_t bf16_tile_bytes = 32 * 32 * 2;
    const uint32_t num_chunks = d_tiles / chunk_tiles;

    for (uint32_t u = unit_start; u < unit_start + units; ++u) {
        const uint32_t block = u / hc;
        const uint32_t h = u - block * hc;
        const uint32_t out_row = (h * blocks + block) * d_tiles;

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t c0 = chunk * chunk_tiles;
            cb_wait_front(cb_out, chunk_tiles);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t s = 0; s < chunk_tiles; ++s) {
                noc_async_write(src + s * bf16_tile_bytes, out.get_noc_addr(out_row + c0 + s), bf16_tile_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, chunk_tiles);
        }
    }
}
