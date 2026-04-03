// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer: OUTPUT first, STATE second
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t state_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t num_heads = get_compile_time_arg_val(0);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t state_tiles_per_head = get_compile_time_arg_val(2);
    constexpr auto output_acc_args = TensorAccessorArgs<3>();
    constexpr auto state_acc_args = TensorAccessorArgs<output_acc_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_17;
    uint32_t output_tile_bytes = get_tile_size(cb_out);
    uint32_t state_tile_bytes = get_tile_size(cb_state_out);
    const auto s_output = TensorAccessor(output_acc_args, output_addr, output_tile_bytes);
    const auto s_state = TensorAccessor(state_acc_args, state_addr, state_tile_bytes);
    for (uint32_t head = 0; head < num_heads; head++) {
        uint32_t out_start = head * v_tiles;
        for (uint32_t vt = 0; vt < v_tiles; vt++) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(out_start + vt, s_output, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
        uint32_t st_start = head * state_tiles_per_head;
        for (uint32_t st = 0; st < state_tiles_per_head; st++) {
            cb_wait_front(cb_state_out, 1);
            noc_async_write_tile(st_start + st, s_state, get_read_ptr(cb_state_out));
            noc_async_write_barrier();
            cb_pop_front(cb_state_out, 1);
        }
    }
}
