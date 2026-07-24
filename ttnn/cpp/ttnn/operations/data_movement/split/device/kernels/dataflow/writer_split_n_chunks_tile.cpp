// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Generalized TILE-layout split writer: each core writes to exactly one output chunk.
// The program factory assigns each core to the correct chunk and passes the right output address.

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(1);

    // WRITER COMPILE TIME ARGS:
    //   arg0 = per_core_tiles_x from factory → HEIGHT tiles per core (j, outer/slow loop)
    //   arg1 = per_core_tiles_y from factory → WIDTH  tiles per core (i, inner/fast loop)
    // Note: the "_y/_x" suffix on the variable names refers to the TENSOR dimension
    // (dim-2 = Y/height, dim-3 = X/width), not the core-grid axis.
    constexpr uint32_t out_num_tiles_per_tensor_y = get_compile_time_arg_val(0);  // HEIGHT, j loop
    constexpr uint32_t out_num_tiles_per_tensor_x = get_compile_time_arg_val(1);  // WIDTH,  i loop
    constexpr uint32_t z = get_compile_time_arg_val(2);
    constexpr uint32_t z_stride = get_compile_time_arg_val(3);
    constexpr uint32_t y_stride = get_compile_time_arg_val(4);
    // One shared TensorAccessorArgs — all output chunks have the same buffer type and page size.
    constexpr auto out_tensor_args = TensorAccessorArgs<5>();

    constexpr uint32_t dfb_id_out0 = 0;
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(out_tensor_args, out_tensor_addr);
    Noc noc;
    DataflowBuffer dfb_out(dfb_id_out0);
    const uint32_t single_tile_size_bytes = dfb_out.get_entry_size();

    uint32_t z_stride_cum = 0;
    for (uint32_t k = 0; k < z; k++) {
        uint32_t y_stride_cum = 0;
        for (uint32_t j = 0; j < out_num_tiles_per_tensor_y; j++) {
            for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
                uint32_t tile_id = y_stride_cum + z_stride_cum + i;
                dfb_out.wait_front(onetile);
                noc.async_write(
                    dfb_out, s, single_tile_size_bytes, {.offset_bytes = 0}, {.page_id = tile_id + out_tensor_tile_id});
                noc.async_write_barrier();
                dfb_out.pop_front(onetile);
            }
            y_stride_cum += y_stride;
        }
        z_stride_cum += z_stride;
    }
}
