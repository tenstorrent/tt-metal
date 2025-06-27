// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/pause.h"

#include <cstdint>

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

void kernel_main() {
    // Runtime args
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(3);
    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(7);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(8);

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint32_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

    // Output tensor config
    const uint32_t value_tensor_tile_size_bytes = get_tile_size(value_tensor_cb_index);
    const DataFormat value_tensor_data_format = get_dataformat(value_tensor_cb_index);
    const InterleavedAddrGenFast<value_tensor_is_dram> output_tensor_accessor = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = value_tensor_tile_size_bytes,
        .data_format = value_tensor_data_format};
    DPRINT << "WRITER: Starting" << ENDL();  // TODO: remove
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            DPRINT << "WRITER: Generating index tile: " << w << ENDL();  // TODO: Remove
            generate_index_tile(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            // PAUSE(); // TODO: Remove
        }  // w loop

        for () {
        }

        // Write value tensor to DRAM
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_wait_front(value_tensor_cb_index, one_tile);
            DPRINT << "WRITER: Writing tile: " << w << " at h: " << h << ENDL();  // TODO: remove
            const uint32_t l1_write_addr_val = get_read_ptr(value_tensor_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
            noc_async_write_tile(tile_offset, output_tensor_accessor, l1_write_addr_val);
            noc_async_write_barrier();
            cb_pop_front(value_tensor_cb_index, one_tile);
        }  // Wt loop
    }  // h loop
    DPRINT << "WRITER: Finished reading and sorting tiles." << ENDL();  // TODO: remove
}
