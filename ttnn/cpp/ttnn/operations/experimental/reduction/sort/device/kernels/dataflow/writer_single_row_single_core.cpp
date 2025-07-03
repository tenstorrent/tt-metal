// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    constexpr uint32_t one_tile = 1;
    // Reserve space
    cb_reserve_back(cb_id, one_tile);

    // Writer config
    const uint32_t writer_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(writer_addr);
    const uint16_t wt_offset = wt << 5;  // wt * 2^(5)

    // Writer loop
    uint32_t count = 0;
    /*
    The 32x32 tile is subdivided into four 16x16 quadrants(faces): top-left, top-right, bottom-left, and bottom-right.
    These quadrants are stored contiguously in memory. Therefore, indices must be written in memory according
    to their respective quadrant, rather than sequentially from left to right across the entire tile.
    */
    constexpr uint32_t tile_faces = 2;
    constexpr uint32_t face_size = 16;
    for (uint32_t i = 0; i < tile_faces; ++i) {
        for (uint32_t j = 0; j < tile_faces; ++j) {
            for (uint32_t k = 0; k < face_size; ++k) {
                for (uint32_t l = 0; l < face_size; l++) {
                    const uint16_t value = l + face_size * j + wt_offset;
                    ptr[count] = value;
                    count++;
                }  // l loop
            }  // k loop
        }  // j loop
    }  // i loop

    // Push the tile
    cb_push_back(cb_id, one_tile);
}

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
    const uint32_t value_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(6);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(7);

    // Output tensor config
    constexpr uint32_t one_tile = 1;
    const uint32_t value_tensor_tile_size_bytes = get_tile_size(value_tensor_cb_index);
    const DataFormat value_tensor_data_format = get_dataformat(value_tensor_cb_index);
    const InterleavedAddrGenFast<value_tensor_is_dram> interleaved_accessor0 = {
        .bank_base_address = value_tensor_buffer_addr,
        .page_size = value_tensor_tile_size_bytes,
        .data_format = value_tensor_data_format};

    // Move data from L1 to DRAMs
    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Generate index tiles
        for (uint32_t w = 0; w < Wt; w++) {
            generate_index_tile(index_tensor_cb_index, w);
        }  // Wt loop

        // Write value tensor to DRAM
        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(value_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_val = get_read_ptr(value_tensor_cb_index);
            noc_async_write_tile(h * Wt + w, interleaved_accessor0, l1_write_addr_val);
            noc_async_write_barrier();
            cb_pop_front(value_tensor_cb_index, one_tile);
        }  // Wt loop
    }  // core_loop_count loop
}
