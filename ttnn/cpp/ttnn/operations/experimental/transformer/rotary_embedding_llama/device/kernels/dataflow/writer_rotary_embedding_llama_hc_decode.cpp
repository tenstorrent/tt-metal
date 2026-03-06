// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for rotary_embedding_llama in HC-transpose decode mode.
//
// Output tensor layout: [1, num_heads, batch_size, head_dim] (interleaved, tilized)
//   Same layout as input — no transpose of output.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t head_start = get_arg_val<uint32_t>(argrt++);
    uint32_t head_end = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t batch_t = get_compile_time_arg_val(1);  // total batch tiles
    constexpr uint32_t Wt = get_compile_time_arg_val(2);       // head_dim_t
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // Output tile linear index: out[0, h, bt, w] = h * batch_t * Wt + bt * Wt + w
    for (uint32_t h = head_start; h < head_end; ++h) {
        for (uint32_t bt = batch_t_start; bt < batch_t_end; ++bt) {
            const uint32_t output_base = h * batch_t * Wt + bt * Wt;

            cb_wait_front(cb_id_out, Wt);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            for (uint32_t w = 0; w < Wt; ++w) {
                noc_async_write_tile(output_base + w, s, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, Wt);
        }
    }
}
