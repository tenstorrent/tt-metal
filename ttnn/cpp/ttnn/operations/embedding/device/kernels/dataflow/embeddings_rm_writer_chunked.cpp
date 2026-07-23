// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    auto num_sticks = get_arg(args::num_sticks);
    auto start_id = get_arg(args::start_id);

    constexpr auto chunk_size = get_arg(args::chunk_size);
    constexpr auto num_chunks = get_arg(args::num_chunks);
    constexpr auto last_chunk_size = get_arg(args::last_chunk_size);

    const auto s0 = TensorAccessor(tensor::output);

    DataflowBuffer dfb_out0(dfb::out0);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t row = start_id; row < end_id; ++row) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            dfb_out0.wait_front(1);
            uint32_t l1_read_addr = dfb_out0.get_read_ptr();
            uint32_t write_size = (chunk < num_chunks - 1) ? chunk_size : last_chunk_size;
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                s0,
                write_size,
                {},
                {.page_id = row, .offset_bytes = chunk * chunk_size});
            noc.async_write_barrier();
            dfb_out0.pop_front(1);
        }
    }
}
