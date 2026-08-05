// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Generalized TILE-layout split writer: each core writes to exactly one output chunk.
// The program factory assigns each core to the correct chunk and binds the right output tensor.

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg(args::out_tensor_tile_id);

    // WRITER COMPILE TIME ARGS:
    //   out_num_tiles_per_tensor_y = per_core_tiles_x from factory → HEIGHT tiles per core (j, outer/slow loop)
    //   out_num_tiles_per_tensor_x = per_core_tiles_y from factory → WIDTH  tiles per core (i, inner/fast loop)
    // Note: the "_y/_x" suffix on out_num_tiles_per_tensor variables refers to the TENSOR dimension
    // (dim-2 = Y/height, dim-3 = X/width). The factory names its per_core_tiles_x/per_core_tiles_y
    // by the CORE-GRID axis instead, so the two conventions are transposed: per_core_tiles_x spreads
    // dim-2 (height) across the x-cores, so the height count arrives here as out_num_tiles_per_tensor_y,
    // and per_core_tiles_y (the width count) as out_num_tiles_per_tensor_x.
    constexpr auto out_num_tiles_per_tensor_y = get_arg(args::out_num_tiles_per_tensor_y);  // HEIGHT, j loop
    constexpr auto out_num_tiles_per_tensor_x = get_arg(args::out_num_tiles_per_tensor_x);  // WIDTH,  i loop
    constexpr auto z = get_arg(args::z);
    constexpr auto z_stride = get_arg(args::z_stride);
    constexpr auto y_stride = get_arg(args::y_stride);

    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::out);
    Noc noc;
    DataflowBuffer dfb_out(dfb::src0);
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
