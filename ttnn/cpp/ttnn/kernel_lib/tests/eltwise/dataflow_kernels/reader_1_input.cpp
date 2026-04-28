// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// 1-input interleaved DRAM reader for eltwise helper tests.
// Compile-time: TensorAccessorArgs blob for the input.
// Runtime:      [src0_addr, num_tiles, start_id].

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr auto s0_args = TensorAccessorArgs<0>();

    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    const uint32_t b0 = get_tile_size(tt::CBIndex::c_0);
    const auto a0 = TensorAccessor(s0_args, src0_addr, b0);

    for (uint32_t t = start_id; t < start_id + num_tiles; ++t) {
        cb0.reserve_back(1);
        noc.async_read(a0, cb0, b0, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb0.push_back(1);
    }
}
