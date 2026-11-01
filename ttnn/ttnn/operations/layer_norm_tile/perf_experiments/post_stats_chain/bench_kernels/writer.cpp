// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// post_stats_chain bench writer: drains the output (R * Wt tiles) per iteration, writes the last one.
// Drains per DEST-lane group so the writer never holds the packer back (output CB is one iteration deep).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t iterations = get_compile_time_arg_val(2);
    constexpr uint32_t io_bytes = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out_acc = TensorAccessor(out_args, get_arg_val<uint32_t>(0), io_bytes);

    DataflowBuffer out(cb_out);
    for (uint32_t it = 0; it < iterations; ++it) {
        out.wait_front(row_tiles);
        if (it + 1 == iterations) {
            for (uint32_t t = 0; t < row_tiles; ++t) {
                noc_async_write(out.get_read_ptr() + t * io_bytes, out_acc.get_noc_addr(t), io_bytes);
            }
            noc_async_write_barrier();
        }
        out.pop_front(row_tiles);
    }
}
