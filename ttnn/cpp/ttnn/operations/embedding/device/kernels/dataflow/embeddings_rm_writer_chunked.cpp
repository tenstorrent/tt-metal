// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t last_chunk_size = get_compile_time_arg_val(4);
    constexpr auto dst0_args = TensorAccessorArgs<5>();

    const auto s0 = TensorAccessor(dst0_args, dst_addr, output_page_size);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t row = start_id; row < end_id; ++row) {
        uint64_t row_base_addr = s0.get_noc_addr(row);
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_id_out0, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            uint32_t write_size = (chunk < num_chunks - 1) ? chunk_size : last_chunk_size;
            uint64_t dst_noc_addr = row_base_addr + chunk * chunk_size;
            noc_async_write(l1_read_addr, dst_noc_addr, write_size);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, 1);
        }
    }
}
