// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated normalize/affine bench writer: drains every rep, stores rep 0 only.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t reps = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out_acc = TensorAccessor(out_args, get_arg_val<uint32_t>(0), out_tile_bytes);

    DataflowBuffer out(cb_out);
    for (uint32_t rep = 0; rep < reps; ++rep) {
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            out.wait_front(1);
            if (rep == 0) {
                noc_async_write(out.get_read_ptr(), out_acc.get_noc_addr(tile), out_tile_bytes);
                noc_async_write_barrier();
            }
            out.pop_front(1);
        }
    }
}
