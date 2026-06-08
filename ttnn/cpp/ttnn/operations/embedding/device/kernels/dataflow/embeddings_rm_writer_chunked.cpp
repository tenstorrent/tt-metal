// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

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

    CircularBuffer cb_out0(cb_id_out0);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t row = start_id; row < end_id; ++row) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_out0.wait_front(1);
            uint32_t l1_read_addr = cb_out0.get_read_ptr();
            uint32_t write_size = (chunk < num_chunks - 1) ? chunk_size : last_chunk_size;
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                s0,
                write_size,
                {},
                {.page_id = row, .offset_bytes = chunk * chunk_size});
            noc.async_write_barrier();
            cb_out0.pop_front(1);
        }
    }
}
