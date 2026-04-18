// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    const uint32_t num_batches = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles_per_batch = get_arg_val<uint32_t>(4);

    if (num_tiles == 0) {
        return;
    }

    const uint32_t tile_size = get_tile_size(cb_dst);
    const auto dst_tensor = TensorAccessor(dst_args, dst_addr, tile_size);

    uint64_t dst_noc_addr = dst_tensor.get_noc_addr(tile_ofs);
    uint32_t dst_noc_ofs = 0;

    const uint32_t large_chunk = num_batches * num_tiles_per_batch;
    uint32_t remaining = num_tiles;

    while (remaining > 0) {
        uint32_t n_tiles_proc;
        if (remaining >= large_chunk) {
            n_tiles_proc = large_chunk;
        } else if (remaining >= num_tiles_per_batch) {
            n_tiles_proc = num_tiles_per_batch;
        } else {
            n_tiles_proc = remaining;
        }

        cb_wait_front(cb_dst, n_tiles_proc);
        uint32_t l1_read_addr = get_read_ptr(cb_dst);

        // TODO: Can we send n_tiles_proc for one noc_async_write transaction?
        for (uint32_t k = 0; k < n_tiles_proc; k++) {
            noc_async_write(l1_read_addr + k * tile_size, dst_noc_addr + dst_noc_ofs, tile_size);
            dst_noc_ofs += tile_size;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_dst, n_tiles_proc);
        remaining -= n_tiles_proc;
    }
}
