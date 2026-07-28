// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for register-based argmax over a non-HW dim (NC-style).
// Writes UINT32 output tiles produced by the compute kernel from cb_out0 to
// the interleaved output buffer (DRAM or L1).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out0 = 16;
    constexpr uint32_t onetile = 1;

    constexpr auto output_tensor_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(output_tensor_args, output_addr);

    Noc noc;
    CircularBuffer cb(cb_out0);
    const uint32_t tile_bytes = get_tile_size(cb_out0);

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        const uint32_t write_tile_id = start_id + out_i;
        cb.wait_front(onetile);
        noc.async_write(
            use<CircularBuffer::AddrSelector::READ_PTR>(cb),
            s0,
            tile_bytes,
            {.offset_bytes = 0},
            {.page_id = write_tile_id});
        noc.async_write_barrier();
        cb.pop_front(onetile);
    }
}
