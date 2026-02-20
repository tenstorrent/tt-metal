// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor_args.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // Compile time args
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr auto dst_tensor_args = TensorAccessorArgs<2>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_responsible = get_arg_val<uint32_t>(2);

    // Create tensor accessor for output
    const auto dst_accessor = TensorAccessor(dst_tensor_args, dst_addr, tile_size_bytes);

    uint32_t tile_id = start_tile_id;

    // Write tiles sequentially from CB to output buffer
    for (uint32_t i = 0; i < num_tiles_responsible; i++) {
        cb_wait_front(cb_id_in, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);

        noc_async_write_tile(tile_id, dst_accessor, l1_read_addr);

        noc_async_write_barrier();

        cb_pop_front(cb_id_in, 1);

        tile_id++;
    }
}
