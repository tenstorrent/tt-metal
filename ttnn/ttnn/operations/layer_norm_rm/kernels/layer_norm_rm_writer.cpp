// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t output_acc_idx = 4;
    constexpr auto output_args = TensorAccessorArgs<output_acc_idx>();

    uint32_t arg_idx = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t N = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_stick_id = get_arg_val<uint32_t>(arg_idx++);

    if (N == 0) {
        return;
    }

    const auto output_accessor = TensorAccessor(output_args, dst_addr, output_stick_size);

    for (uint32_t tr = 0; tr < N; tr++) {
        cb_wait_front(cb_id_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        for (uint32_t s = 0; s < tile_height; s++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(start_stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            start_stick_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, Wt);
    }
}
