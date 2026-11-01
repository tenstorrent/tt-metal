// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// stats_fusion bench writer. MODE 0: drains y = c * inv_std (Wt tiles) per iteration, writes the last.
// MODE 1: drains c (Wt tiles) and inv_std (1 tile) per iteration, writes the last.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_main = get_compile_time_arg_val(0);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t iterations = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t mode = get_compile_time_arg_val(5);
    constexpr auto main_args = TensorAccessorArgs<6>();
    constexpr auto inv_args = TensorAccessorArgs<main_args.next_compile_time_args_offset()>();

    const uint32_t main_addr = get_arg_val<uint32_t>(0);
    const uint32_t inv_addr = get_arg_val<uint32_t>(1);
    const auto main_acc = TensorAccessor(main_args, main_addr, tile_bytes);
    const auto inv_acc = TensorAccessor(inv_args, inv_addr, tile_bytes);

    DataflowBuffer main_cb(cb_main);
    DataflowBuffer inv_cb(cb_inv_std);
    for (uint32_t it = 0; it < iterations; ++it) {
        const bool last = it + 1 == iterations;
        main_cb.wait_front(Wt);
        if constexpr (mode == 1) {
            inv_cb.wait_front(1);
        }
        if (last) {
            for (uint32_t t = 0; t < Wt; ++t) {
                noc_async_write(main_cb.get_read_ptr() + t * tile_bytes, main_acc.get_noc_addr(t), tile_bytes);
            }
            if constexpr (mode == 1) {
                noc_async_write(inv_cb.get_read_ptr(), inv_acc.get_noc_addr(0), tile_bytes);
            }
            noc_async_write_barrier();
        }
        main_cb.pop_front(Wt);
        if constexpr (mode == 1) {
            inv_cb.pop_front(1);
        }
    }
}
