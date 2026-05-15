// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_dst = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_ofs = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles_per_batch = get_arg_val<uint32_t>(3);
    const uint32_t noc = get_arg_val<uint32_t>(4);

    if (num_tiles == 0) {
        return;
    }

    const uint32_t tile_size = get_tile_size(cb_dst);
    const auto dst_tensor = TensorAccessor(dst_args, dst_addr, tile_size);

    uint64_t dst_noc_addr = dst_tensor.get_noc_addr(tile_ofs, /*offset=*/0, noc);
    uint32_t dst_noc_ofs = 0;

    uint32_t remaining = num_tiles;
    while (remaining > 0) {
        uint32_t n_tiles_proc;
        if (remaining >= num_tiles_per_batch) {
            n_tiles_proc = num_tiles_per_batch;
        } else {
            n_tiles_proc = remaining;
        }

        cb_wait_front(cb_dst, n_tiles_proc);
        uint32_t l1_read_addr = get_read_ptr(cb_dst);

        for (uint32_t k = 0; k < n_tiles_proc; k++) {
            noc_async_write(l1_read_addr + k * tile_size, dst_noc_addr + dst_noc_ofs, tile_size, noc);
            dst_noc_ofs += tile_size;
        }
        noc_async_write_barrier(noc);

        cb_pop_front(cb_dst, n_tiles_proc);
        remaining -= n_tiles_proc;
    }
}
