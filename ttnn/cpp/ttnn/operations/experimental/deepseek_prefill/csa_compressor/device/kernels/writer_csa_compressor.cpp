// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t pooled_cb = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    const auto output = TensorAccessor(output_args, output_addr);
    CircularBuffer pooled(pooled_cb);
    Noc noc;
    constexpr uint32_t tile_bytes = 2048;
    for (uint32_t tile = 0; tile < output_tiles; ++tile) {
        pooled.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(pooled), output, tile_bytes, {}, {.page_id = tile});
        noc.async_write_barrier();
        pooled.pop_front(1);
    }
}
