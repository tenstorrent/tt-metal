// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Sequential writer: drains `num_tiles` from the compute output CB to the
// interleaved output tensor starting at tile index `start_id`.
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = 2;
    constexpr uint32_t onetile = 1;

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_out, onetile);
        uint32_t l1_read = get_read_ptr(cb_out);
        noc_async_write_tile(start_id + i, dst, l1_read);
        noc_async_write_barrier();
        cb_pop_front(cb_out, onetile);
    }
}
