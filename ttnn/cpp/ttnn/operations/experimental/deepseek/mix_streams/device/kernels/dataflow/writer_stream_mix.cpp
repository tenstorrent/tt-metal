// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Writes each work unit's n_out tiles to output stream i's copy of the unit's tile position.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out_id = get_compile_time_arg_val(0);
    constexpr uint32_t n_out = get_compile_time_arg_val(1);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t t_tiles = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_out(cb_out_id);

    constexpr uint32_t stream_tiles = t_tiles * d_tiles;
    const uint32_t tile_bytes = cb_out.get_tile_size();

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        cb_out.wait_front(n_out);
        for (uint32_t i = 0; i < n_out; ++i) {
            noc.async_write(
                cb_out, out, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = i * stream_tiles + unit});
        }
        noc.async_write_barrier();
        cb_out.pop_front(n_out);
    }
}
