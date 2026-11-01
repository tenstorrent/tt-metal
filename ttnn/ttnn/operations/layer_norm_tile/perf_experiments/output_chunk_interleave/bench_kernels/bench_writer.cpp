// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated output-stage bench writer: the op's store pattern (wait C tiles, write, flush, pop). Every rep
// drains; rep 0 writes, later reps only pop (their data is not meaningful).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t reps = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t chunk = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    const auto out_acc = TensorAccessor(out_args, get_arg_val<uint32_t>(0), out_tile_bytes);

    DataflowBuffer out(cb_out);
    for (uint32_t rep = 0; rep < reps; ++rep) {
        for (uint32_t tile = 0; tile < num_tiles; tile += chunk) {
            out.wait_front(chunk);
            if (rep == 0) {
                const uint32_t base = out.get_read_ptr();
                for (uint32_t j = 0; j < chunk; ++j) {
                    noc_async_write(base + j * out_tile_bytes, out_acc.get_noc_addr(tile + j), out_tile_bytes);
                }
                noc_async_writes_flushed();
            }
            out.pop_front(chunk);
        }
    }
    noc_async_write_barrier();
}
