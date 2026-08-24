// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr std::uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    std::uint32_t dst_addr = get_arg_val<std::uint32_t>(0);
    std::uint32_t start_id = get_arg_val<std::uint32_t>(1);
    std::uint32_t num_tiles = get_arg_val<std::uint32_t>(2);
    std::uint32_t end_id = start_id + num_tiles;

    const auto output_addrg = TensorAccessor(dst_args, dst_addr);

    const std::uint32_t page_bytes = get_local_cb_interface(output_cb_id).fifo_page_size;

    Noc noc;
    CircularBuffer cb_output(output_cb_id);

    for (std::uint32_t i = start_id; i < end_id; ++i) {
        cb_output.wait_front(1);
        std::uint32_t output_cb_read_ptr = cb_output.get_read_ptr();
        noc.async_write(CoreLocalMem<std::uint32_t>(output_cb_read_ptr), output_addrg, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb_output.pop_front(1);
    }
    noc.async_write_barrier();
}
