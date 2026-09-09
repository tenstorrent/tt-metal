// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// #define DEBUG
#ifdef DEBUG
// #include "api/debug/dprint.h"
#endif

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);

    // COMPILE TIME ARGS
    constexpr auto z = get_arg(args::z);
    constexpr auto out_num_tiles_per_tensor_y = get_arg(args::out_num_tiles_per_tensor_y);
    constexpr auto out_num_tiles_per_tensor_x = get_arg(args::out_num_tiles_per_tensor_x);
    constexpr auto z_stride = get_arg(args::z_stride);
    constexpr auto y_stride = get_arg(args::y_stride);

    constexpr uint32_t out_num_tensors = 1;

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::in0);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::src0);
    const uint32_t single_tile_size_bytes = dfb_in0.get_entry_size();

    uint32_t tensor_stride = out_num_tiles_per_tensor_x;
    uint32_t tensor_stride_cum = 0;
#ifdef DEBUG
    // DPRINT("Reader Tile ID Offset: {}\n\n", in0_tensor_tile_id);
    // DPRINT("Reader Z Stride: {}\n", z_stride);
    // DPRINT("Reader Y Stride: {}\n", y_stride);
#endif
    for (uint32_t out_tensor_id = 0; out_tensor_id < out_num_tensors; out_tensor_id++) {
        uint32_t z_stride_cum = 0;
        for (uint32_t k = 0; k < z; k++) {
            uint32_t y_stride_cum = 0;
            for (uint32_t j = 0; j < out_num_tiles_per_tensor_y; j++) {
                for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
                    uint32_t tile_id = y_stride_cum + tensor_stride_cum + z_stride_cum + i;
                    dfb_in0.reserve_back(onetile);
                    noc.async_read(
                        s0,
                        dfb_in0,
                        single_tile_size_bytes,
                        {.page_id = tile_id + in0_tensor_tile_id},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    dfb_in0.push_back(onetile);
#ifdef DEBUG
                    // DPRINT("Reader Tile ID: {}\n", tile_id + in0_tensor_tile_id);
                    // DPRINT("Reader Address: {}\n\n", l1_write_addr_in0);
#endif
                }
                y_stride_cum += y_stride;
            }
            z_stride_cum += z_stride;
        }

        tensor_stride_cum += tensor_stride;
    }
}
