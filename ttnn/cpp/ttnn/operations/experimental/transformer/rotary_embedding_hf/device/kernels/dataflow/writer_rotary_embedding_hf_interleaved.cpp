// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t output_tile_bytes = get_tile_size(output_cb_id);
    const auto s = TensorAccessor(dst_args, dst_addr, output_tile_bytes);

    CircularBuffer cb_output(output_cb_id);

    uint32_t output_curr_id = start_id;

#ifdef OUT_SHARDED
    cb_output.wait_front(num_tiles);
#else
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_output.wait_front(1);
        uint32_t l1_read_addr = cb_output.get_read_ptr();
        noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, output_tile_bytes, {}, {.page_id = output_curr_id});
        noc.async_write_barrier();
        cb_output.pop_front(1);
        output_curr_id++;
    }
#endif
}
