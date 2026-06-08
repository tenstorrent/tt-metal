// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_core_data_exchange_common.hpp"
#include "sort_dataflow_common.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_peer_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t physical_core_lookup_table_cb_index =
        get_compile_time_arg_val(5);  // unused - for future improvements
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t Ht = get_compile_time_arg_val(7);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(9);           // unused - for future improvements
    const uint32_t sem_exchange_addr = get_semaphore(get_compile_time_arg_val(10));  // unused - for future improvements
    constexpr bool is_32_bit_data = get_compile_time_arg_val(11) == 1;
    constexpr bool is_row_major = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t rm_value_output_cb_index = get_compile_time_arg_val(13);
    constexpr uint32_t W_value_slice_bytes = get_compile_time_arg_val(14);

    constexpr auto value_tensor_args = TensorAccessorArgs<15>();

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_pair_start = core_id * number_of_pairs_processed_by_each_core;
    const uint16_t processing_pair_end = processing_pair_start + number_of_pairs_processed_by_each_core;

    // Output tensor config
    const auto output_tensor_accessor = TensorAccessor(value_tensor_args, output_tensor_buffer_addr);

    Noc noc;
    CircularBuffer value_tensor_cb(value_tensor_cb_index);
    CircularBuffer rm_value_output_cb(rm_value_output_cb_index);
    CircularBuffer physical_core_lookup_table_cb(physical_core_lookup_table_cb_index);
    constexpr uint32_t value_tensor_tile_size = get_tile_size(value_tensor_cb_index);

    constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT
    const uint32_t value_slice_offset_bytes = core_id * W_value_slice_bytes;

    for (uint32_t h = 0; h < Ht; h++) {
        // Generate input index tiles (TILE format).
        // The RM path also relies on these — the compute kernel sorts indices
        // alongside values in TILE format, then pack_untilize's the result into
        // RM rows for the reader to drain.  No need for an RM-specific index
        // generator here.
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            } else {
                generate_index_tile<uint16_t>(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            }
        }  // w loop

        if constexpr (is_row_major) {
            // ROW_MAJOR output values: drain TILE_H untilized value rows from
            // rm_value_output_cb (compute pack_untilize'd them) and write each
            // row's per-core W-slice back to DRAM.
            const uint32_t row_base = h * TILE_H;
            for (uint32_t row = 0; row < TILE_H; row++) {
                rm_value_output_cb.wait_front(one_tile);
                noc.async_write(
                    rm_value_output_cb,
                    output_tensor_accessor,
                    W_value_slice_bytes,
                    {.offset_bytes = 0},
                    {.page_id = row_base + row, .offset_bytes = value_slice_offset_bytes});
                noc.async_write_barrier();
                rm_value_output_cb.pop_front(one_tile);
            }
        } else {
            // Write value tensor to DRAM (TILE path)
            for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
                value_tensor_cb.wait_front(one_tile);
                const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
                noc.async_write(
                    value_tensor_cb,
                    output_tensor_accessor,
                    value_tensor_tile_size,
                    {.offset_bytes = 0},
                    {.page_id = tile_offset, .offset_bytes = 0});
                noc.async_write_barrier();
                value_tensor_cb.pop_front(one_tile);
            }  // Wt loop
        }
    }  // h loop
    physical_core_lookup_table_cb.push_back(one_tile);
}
