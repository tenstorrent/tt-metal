// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Sparse MoE writer: writes output tiles from compute to DRAM.
// Output: (1, 1, batch, expert_width_per_core) per core, written to the correct
// column offset in the full (1, 1, batch, total_expert_width) output tensor.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t tile_start = get_arg_val<uint32_t>(1);  // first output tile for this core
    uint32_t num_tiles = get_arg_val<uint32_t>(2);   // total output tiles for this core

    constexpr auto output_acc_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    uint32_t tile_bytes = get_tile_size(cb_out);

    const auto s_output = TensorAccessor(output_acc_args, output_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(tile_start + t, s_output, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
