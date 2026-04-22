// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(src_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in(cb_id_in0);

    uint32_t end_page_id = start_page_id + num_tiles;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_in.reserve_back(1);
        noc.async_read(s, cb_in, tile_bytes, {.page_id = page_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
