// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for scaled_dot_product_attention (Flash Attention).
//
// Stage 14 (output): Drains cb_out (the final normalized output O / l_i) from
// L1 to DRAM. After all KV-blocks for a Q-block, the compute kernel produces
// the final normalized output in cb_out (B_q_t * D_t tiles). The writer streams
// these tiles to the output DRAM buffer in tile-row-major order.
//
// The writer uses TensorAccessor to map tile indices to DRAM page addresses.
// CT args: [num_o_tiles, ...TensorAccessorArgs(output)]
// RT args: [output_addr, start_id]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args: [output_addr, start_id]
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);

    // Compile-time args: [num_o_tiles, ...TensorAccessorArgs]
    constexpr uint32_t num_o_tiles = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_out = tt::CBIndex::c_17;
    uint32_t tile_bytes = get_tile_size(cb_out);
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // Drain cb_out to DRAM in tile-row-major order.
    // The compute kernel pushes B_q_t * D_t tiles; we wait for each, write it
    // to DRAM, then pop to free the CB page for the next tile.
    for (uint32_t i = start_id; i < start_id + num_o_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
