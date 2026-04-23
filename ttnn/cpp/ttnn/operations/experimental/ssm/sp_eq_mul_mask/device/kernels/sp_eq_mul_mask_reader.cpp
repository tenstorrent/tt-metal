// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Sequential reader for two same-shape interleaved input tensors.
// Each core reads `num_tiles` tiles from [start_id, start_id + num_tiles) of
// both A and B, pushing each into its respective circular buffer.
void kernel_main() {
    const uint32_t src_a_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t onetile = 1;

    constexpr auto src_a_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_a_args.next_compile_time_args_offset()>();
    const auto src_a = TensorAccessor(src_a_args, src_a_addr);
    const auto src_b = TensorAccessor(src_b_args, src_b_addr);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_a, onetile);
        uint32_t l1_a = get_write_ptr(cb_a);
        noc_async_read_tile(start_id + i, src_a, l1_a);

        cb_reserve_back(cb_b, onetile);
        uint32_t l1_b = get_write_ptr(cb_b);
        noc_async_read_tile(start_id + i, src_b, l1_b);

        noc_async_read_barrier();
        cb_push_back(cb_a, onetile);
        cb_push_back(cb_b, onetile);
    }
}
