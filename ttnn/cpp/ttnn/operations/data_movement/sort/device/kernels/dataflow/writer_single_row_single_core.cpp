// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include "sort_dataflow_common.hpp"

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input value data from DRAM and writes it to L1 circular buffer.
    * Write processed index data from L1 to DRAM.

Writer:
    * Generates index input data and writes it to L1 circular buffer.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t core_loop_count = get_arg(args::core_loop_count);

    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t compute_with_storage_grid_size_y = get_arg(args::compute_with_storage_grid_size_y);

    // is_32_bit_data - true when fp32_dest_acc_en is enabled (float32 input or
    // uint32 index). Both TILE and ROW_MAJOR paths must generate uint32 index tiles
    // when the sort kernel runs in 32-bit DEST mode.
    constexpr bool is_32_bit_data = get_arg(args::is_32_bit_data) == 1;
    constexpr uint32_t W_value_bytes = get_arg(args::W_value_bytes);

    constexpr uint32_t one_tile = 1;

    // TensorAccessor handles both interleaved and sharded buffers natively.
    // For TILE layout: one "page" = one tile.
    // For ROW_MAJOR layout: one "page" = one row of W elements (W_value_bytes).
    const auto value_accessor = TensorAccessor(tensor::value_tensor);

    Noc noc;
#ifndef IS_ROW_MAJOR
    DataflowBuffer value_tensor_dfb(dfb::value_tensor);
    const uint32_t value_tensor_tile_size = value_tensor_dfb.get_tile_size();

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Generate index tiles into the index DFB (consumed by compute)
        for (uint32_t w = 0; w < Wt; w++) {
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(dfb::index_tensor, w);
            } else {
                generate_index_tile<uint16_t>(dfb::index_tensor, w);
            }
        }

        // Write sorted value tiles from value_tensor_dfb → DRAM
        for (uint32_t w = 0; w < Wt; w++) {
            value_tensor_dfb.wait_front(one_tile);
            noc.async_write(
                value_tensor_dfb,
                value_accessor,
                value_tensor_tile_size,
                {.offset_bytes = 0},
                {.page_id = h * Wt + w, .offset_bytes = 0});
            noc.async_write_barrier();
            value_tensor_dfb.pop_front(one_tile);
        }
    }
#else
    // ------------------------------------------------------------------
    // ROW_MAJOR path
    //
    // The value accessor's page size = W_value_bytes (one RM row).
    //
    // Per loop iteration we handle one tile-row = 32 consecutive rows:
    //   Input:  generate Wt TILE-format index tiles into the index DFB
    //           (compute kernel sorts them alongside the tilized values).
    //   Output: drain 32 untilized value pages from rm_value_output_dfb
    //           → write via noc.async_write → value DRAM buffer.
    // ------------------------------------------------------------------
    DataflowBuffer rm_value_output_dfb(dfb::rm_value_output);

    constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Generate Wt index tiles (TILE / integer format) into the index DFB.
        // The topk LLK reads indices via LO16 (uint16) or INT32 (uint32) mode, so
        // the index DFB must contain raw unsigned integers, not floating-point values.
        for (uint32_t w = 0; w < Wt; w++) {
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(dfb::index_tensor, w);
            } else {
                generate_index_tile<uint16_t>(dfb::index_tensor, w);
            }
        }

        // Drain 32 sorted RM value rows from rm_value_output_dfb → DRAM
        const uint32_t row_base = h * TILE_H;
        for (uint32_t row = 0; row < TILE_H; row++) {
            rm_value_output_dfb.wait_front(one_tile);
            noc.async_write(
                rm_value_output_dfb,
                value_accessor,
                W_value_bytes,
                {.offset_bytes = 0},
                {.page_id = row_base + row, .offset_bytes = 0});
            noc.async_write_barrier();
            rm_value_output_dfb.pop_front(one_tile);
        }
    }
#endif
}
