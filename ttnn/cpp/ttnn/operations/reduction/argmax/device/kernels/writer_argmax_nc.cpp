// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for register-based argmax over a non-HW dim (NC-style).
// Writes UINT32 output tiles produced by the compute kernel from cb_out0 to
// the interleaved output buffer (DRAM or L1).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out0 = 16;
    constexpr uint32_t onetile = 1;

    constexpr auto output_tensor_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(output_tensor_args, output_addr);

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        const uint32_t write_tile_id = start_id + out_i;
        cb_wait_front(cb_out0, onetile);
        const uint32_t read_ptr = get_read_ptr(cb_out0);
        noc_async_write_tile(write_tile_id, s0, read_ptr);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, onetile);
    }
}
