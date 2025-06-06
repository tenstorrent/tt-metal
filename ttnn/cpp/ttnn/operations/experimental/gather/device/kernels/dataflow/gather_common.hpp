// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstdint>

template <typename T>
FORCE_INLINE uint32_t read_data_from_type(const uint32_t l1_addr, const uint32_t count) {
    volatile tt_l1_ptr T* ptr = reinterpret_cast<volatile tt_l1_ptr T*>(l1_addr);
    return ptr[count];
}

FORCE_INLINE uint32_t get_value_from_tile(
    const uint32_t l1_read_addr, const uint32_t count, const uint32_t input_index_tensor_data_format_size) {
    switch (input_index_tensor_data_format_size) {
        case sizeof(uint8_t): {  // 1
            return read_data_from_type<uint8_t>(l1_read_addr, count);
        }
        case sizeof(uint16_t): {  // 2
            return read_data_from_type<uint16_t>(l1_read_addr, count);
        }
        case sizeof(uint32_t): {  // 4
            return read_data_from_type<uint32_t>(l1_read_addr, count);
        }
        case sizeof(uint64_t): {  // 8
            return read_data_from_type<uint64_t>(l1_read_addr, count);
        }
        default: {
            return read_data_from_type<uint16_t>(l1_read_addr, count);
        }
    }
}

template <typename T>
FORCE_INLINE void write_data_from_type(const uint32_t l1_addr, const uint32_t count, const uint32_t value) {
    volatile tt_l1_ptr T* ptr = reinterpret_cast<volatile tt_l1_ptr T*>(l1_addr);
    ptr[count] = value;
}

FORCE_INLINE void write_value_to_tile(
    const uint32_t l1_read_addr, const uint32_t count, const uint32_t data_format_size, const uint32_t value) {
    switch (data_format_size) {
        case sizeof(uint8_t): {  // 1
            write_data_from_type<uint8_t>(l1_read_addr, count, value);
            break;
        }
        case sizeof(uint16_t): {  // 2
            write_data_from_type<uint16_t>(l1_read_addr, count, value);
            break;
        }
        case sizeof(uint32_t): {  // 4
            write_data_from_type<uint32_t>(l1_read_addr, count, value);
            break;
        }
        case sizeof(uint64_t): {  // 8
            write_data_from_type<uint64_t>(l1_read_addr, count, value);
            break;
        }
        default: {
            write_data_from_type<uint16_t>(l1_read_addr, count, value);
            break;
        }
    }
}

FORCE_INLINE void process_input_tile(
    const uint32_t input_tensor_cb_index,
    const uint32_t input_index_tensor_cb_index,
    const uint32_t output_tensor_cb_index,
    const uint32_t current_processed_input_tile_id,
    const uint32_t tile_width) {
    // Constants
    const uint32_t tile_width_mask = tile_width - 1;

    // Dataformats size
    const uint32_t input_tensor_data_format_size =
        get_tile_size(input_tensor_cb_index) / get_tile_hw(input_tensor_cb_index);
    const uint32_t input_index_tensor_data_format_size =
        get_tile_size(input_index_tensor_cb_index) / get_tile_hw(input_index_tensor_cb_index);
    const uint32_t output_tensor_data_format_size =
        get_tile_size(output_tensor_cb_index) / get_tile_hw(output_tensor_cb_index);

    uint32_t input_tensor_l1_read_addr = get_read_ptr(input_tensor_cb_index);
    const uint32_t input_index_tensor_l1_read_addr = get_read_ptr(input_index_tensor_cb_index);
    const uint32_t output_tensor_l1_read_addr = get_read_ptr(output_tensor_cb_index);

    uint32_t count = 0;
    constexpr uint32_t tile_faces = 2;
    constexpr uint32_t face_size = 16;
    constexpr uint32_t FACE_SIZE_MASK = face_size - 1;

    for (uint32_t i = 0; i < tile_faces; ++i) {
        for (uint32_t j = 0; j < tile_faces; ++j) {
            for (uint32_t k = 0; k < face_size; ++k) {
                for (uint32_t l = 0; l < face_size; l++) {
                    // Read global index
                    const uint32_t global_index = get_value_from_tile(
                        input_index_tensor_l1_read_addr, count, input_index_tensor_data_format_size);

                    // Calculate local index
                    const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);

                    if (tile_idx != current_processed_input_tile_id) {
                        // If the tile index is not the one we are currently processing, skip it
                        count++;
                        continue;
                    }

                    const uint32_t index_in_local_tile = global_index & tile_width_mask;
                    const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                    const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                    const uint16_t local_index =
                        which_row * (face_size * face_size) + k * face_size + which_col + i * (tile_width * face_size);

                    // Read value
                    const uint32_t value =
                        get_value_from_tile(input_tensor_l1_read_addr, local_index, input_tensor_data_format_size);

                    write_value_to_tile(output_tensor_l1_read_addr, count, output_tensor_data_format_size, value);
                    count++;
                }  // l loop
            }  // k loop
        }  // j loop
    }  // i loop
}
