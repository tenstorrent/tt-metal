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
