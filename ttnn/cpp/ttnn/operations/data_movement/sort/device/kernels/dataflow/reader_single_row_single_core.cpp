// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

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
    constexpr uint32_t W_value_bytes = get_arg(args::W_value_bytes);
    constexpr uint32_t W_index_bytes = get_arg(args::W_index_bytes);

    // Input tensor config
    constexpr uint32_t one_tile = 1;

    // TensorAccessors handle both interleaved and sharded buffers natively.
    // For TILE layout: one "page" in the accessor = one tile.
    // For ROW_MAJOR layout: one "page" in the accessor = one row of W elements.
    const auto input_accessor = TensorAccessor(tensor::input_tensor);
    const auto index_accessor = TensorAccessor(tensor::index_tensor);

    Noc noc;
#ifndef IS_ROW_MAJOR
    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_output_dfb(dfb::index_tensor_output);
    const uint32_t input_tensor_tile_size = input_tensor_dfb.get_tile_size();
    const uint32_t index_tensor_tile_size = index_output_dfb.get_tile_size();

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Read input tiles from DRAM → tile input DFB
        for (uint32_t w = 0; w < Wt; w++) {
            input_tensor_dfb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                input_tensor_dfb,
                input_tensor_tile_size,
                {.page_id = h * Wt + w, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_tensor_dfb.push_back(one_tile);
        }

        // Write sorted index tiles from index output DFB → DRAM
        for (uint32_t w = 0; w < Wt; w++) {
            index_output_dfb.wait_front(one_tile);
            noc.async_write(
                index_output_dfb,
                index_accessor,
                index_tensor_tile_size,
                {.offset_bytes = 0},
                {.page_id = h * Wt + w, .offset_bytes = 0});
            noc.async_write_barrier();
            index_output_dfb.pop_front(one_tile);
        }
    }
#else
    // ------------------------------------------------------------------
    // ROW_MAJOR path
    //
    // The input accessor's page size = W_value_bytes (one RM row).
    // The index accessor's page size = W_index_bytes (one RM index row).
    //
    // For each tile-row (TILE_HEIGHT = 32 consecutive logical rows):
    //   Input:  read 32 pages via noc.async_read → rm_input_dfb
    //           so the compute kernel can tilize them.
    //   Output: drain 32 untilized index pages from rm_index_output_dfb
    //           → write via noc.async_write → index DRAM buffer.
    // ------------------------------------------------------------------
    DataflowBuffer rm_input_dfb(dfb::rm_input);
    DataflowBuffer rm_index_output_dfb(dfb::rm_index_output);

    constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Base page index for this tile-row group in the RM buffer
        const uint32_t row_base = h * TILE_H;

        // --- Read TILE_H input rows into rm_input_dfb ---
        for (uint32_t row = 0; row < TILE_H; row++) {
            rm_input_dfb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                rm_input_dfb,
                W_value_bytes,
                {.page_id = row_base + row, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            rm_input_dfb.push_back(one_tile);
        }

        // --- Drain TILE_H untilized index rows from rm_index_output_dfb → DRAM ---
        //
        // Compute kernel pack_untilize'd Wt sorted index tiles into
        // TILE_HEIGHT contiguous RM pages in rm_index_output_dfb.
        // pack_untilize_block writes uint16/uint32 elements in the natural
        // little-endian layout that the host expects, so no byte swap is
        // required here regardless of the index dtype.
        for (uint32_t row = 0; row < TILE_H; row++) {
            rm_index_output_dfb.wait_front(one_tile);
            noc.async_write(
                rm_index_output_dfb,
                index_accessor,
                W_index_bytes,
                {.offset_bytes = 0},
                {.page_id = row_base + row, .offset_bytes = 0});
            noc.async_write_barrier();
            rm_index_output_dfb.pop_front(one_tile);
        }
    }
#endif
}
