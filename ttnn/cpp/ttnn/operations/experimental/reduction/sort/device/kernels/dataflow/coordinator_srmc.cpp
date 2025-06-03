// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

#include "debug/dprint.h"

FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // Constants
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
    const uint32_t start_core_physical_coord_x = get_arg_val<uint32_t>(0);
    const uint32_t start_core_physical_coord_y = get_arg_val<uint32_t>(1);
    const uint32_t end_core_physical_coord_x = get_arg_val<uint32_t>(2);
    const uint32_t end_core_physical_coord_y = get_arg_val<uint32_t>(3);
    const uint32_t coordinator_to_cores_semaphore_id = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t cores_to_coordinator_semaphore_id = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t number_of_dest = get_arg_val<uint32_t>(6);
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(7);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(8);
    const uint32_t output_index_tensor_buffer_addr = get_arg_val<uint32_t>(9);

    // Compile time args
    constexpr uint32_t total_work_units = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(3);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(6);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(7) == 1;
    constexpr bool output_tensor_is_dram = get_compile_time_arg_val(8) == 1;
    constexpr bool output_index_tensor_is_dram = get_compile_time_arg_val(9) == 1;

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_addr_ger = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Output tensor config
    const InterleavedAddrGenFast<output_tensor_is_dram> output_tensor_addr_gen = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Output index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_cb_index);
    const InterleavedAddrGenFast<output_index_tensor_is_dram> output_index_tensor_addr_gen = {
        .bank_base_address = output_index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    // Semaphore setup
    volatile tt_l1_ptr uint32_t* semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cores_to_coordinator_semaphore_id);
    noc_semaphore_set(semaphore_ptr, 0);  // Reset the semaphore
    const uint64_t semaphore_global_multicast_addr = get_noc_multicast_addr(
        start_core_physical_coord_x,
        start_core_physical_coord_y,
        end_core_physical_coord_x,
        end_core_physical_coord_y,
        coordinator_to_cores_semaphore_id);

    const uint32_t number_of_confirmations = Wt / 2;

    // Copy input data to output and generate index tiles
    for (uint32_t h = 0; h < Ht; h++) {
        // Prepare and move data
        for (uint32_t w = 0; w < Wt; w++) {
            // Generate indexes
            generate_index_tile(index_tensor_cb_index, w);

            // Save index tile to output index tensor
            cb_wait_front(index_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_index_tensor_cb = get_read_ptr(index_tensor_cb_index);
            noc_async_write_tile(h * Wt + w, output_index_tensor_addr_gen, l1_write_addr_index_tensor_cb);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_cb_index, one_tile);

            // Read input value data
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_input_tensor_cb = get_write_ptr(input_tensor_cb_index);
            noc_async_read_tile(h * Wt + w, input_tensor_addr_ger, l1_write_addr_input_tensor_cb);
            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);

            // Write output value data
            cb_wait_front(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_output_tensor_cb = get_read_ptr(input_tensor_cb_index);
            noc_async_write_tile(h * Wt + w, output_tensor_addr_gen, l1_write_addr_output_tensor_cb);
            noc_async_write_barrier();
            cb_pop_front(input_tensor_cb_index, one_tile);

        }  // Wt loop

        // Wait until all cores are ready to start
        noc_semaphore_wait(semaphore_ptr, number_of_dest);
        noc_semaphore_set(semaphore_ptr, 0);  // Reset the semaphore

        // Set signal to start processing
        noc_semaphore_set_multicast(coordinator_to_cores_semaphore_id, semaphore_global_multicast_addr, number_of_dest);

        // Calculate sorting stages
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                // Set signal to start processing next sub-stage
                noc_semaphore_set_multicast(
                    coordinator_to_cores_semaphore_id, semaphore_global_multicast_addr, number_of_dest);

                // Wait until cores will process and save data
                noc_semaphore_wait(semaphore_ptr, number_of_confirmations);
                noc_semaphore_set(semaphore_ptr, 0);  // Reset the semaphore
            }  // sub loop
        }  // stage loop
    }  // Ht loop
}
