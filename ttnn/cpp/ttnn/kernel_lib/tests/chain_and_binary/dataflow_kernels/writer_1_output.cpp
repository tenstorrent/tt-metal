// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Minimal single-output interleaved DRAM writer for kernel_lib tests.
// Drains CB c_16 (output) to the destination tensor.
// Runtime args: dst_addr, num_tiles, start_id.
// Compile-time args: [out_cb_id, TensorAccessorArgs...].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_id_out);
    const uint32_t bytes = get_tile_size(cb_id_out);
    const auto acc = TensorAccessor(dst_args, dst_addr, bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t t = start_id; t < start_id + num_tiles; ++t) {
        cb_out.wait_front(onetile);
        noc.async_write(cb_out, acc, bytes, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        cb_out.pop_front(onetile);
    }
}
