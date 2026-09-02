// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t tile_height = 32;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_core = get_arg_val<uint32_t>(2);
    const uint32_t tile_width_size = get_arg_val<uint32_t>(3);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(4);
    uint32_t offset_within_stick = get_arg_val<uint32_t>(5);

    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer cb_out(cb_id_out0);

    uint32_t curr_stick_offset = 0;
    uint32_t row_stick_ids[tile_height];

    auto write_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_out.wait_front(num_tiles);
        for (uint32_t k = 0; k < tile_height; k++) {
            noc.async_write(
                cb_out,
                s,
                width_size,
                {.offset_bytes = k * width_size},
                {.page_id = row_stick_ids[k], .offset_bytes = curr_stick_offset});
        }
        noc.async_write_barrier();
        cb_out.pop_front(num_tiles);
    };

    uint32_t stick_id = start_stick_id;

    uint32_t curr_offset = offset_within_stick;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        for (uint32_t tile_id = 0; tile_id < num_tiles_per_core; tile_id++) {
            for (uint32_t j = 0; j < tile_height; j++) {
                row_stick_ids[j] = stick_id + j;
            }
            curr_stick_offset = curr_offset;
            write_tiles(1, tile_width_size);
            curr_offset += tile_width_size;
        }

        stick_id += tile_height;
    }
}
