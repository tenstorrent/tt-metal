// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for cb_helpers_compute.hpp
// Do not include directly - include cb_helpers_compute.hpp instead

#include "api/compute/cb_api.h"
#include "tt-metalium/circular_buffer_constants.h"

namespace compute_kernel_lib {

constexpr uint32_t DATUM_SHIFT = 10;  // 32x32 = 1024 datums
constexpr uint32_t EXP_SHIFT = 6;    // 64 exponents for block formats

ALWI constexpr uint32_t get_full_tile_size_impl(DataFormat format) {
    switch (format) {
        case DataFormat::UInt8:
        case DataFormat::Lf8:
        case DataFormat::Int8:
            return (1 << DATUM_SHIFT);
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return (1 << (DATUM_SHIFT + 1));
        case DataFormat::Float32:
        case DataFormat::Int32:
        case DataFormat::UInt32:
            return (1 << (DATUM_SHIFT + 2));
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b:
            return (1 << DATUM_SHIFT) + (1 << EXP_SHIFT);
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b:
            return (1 << (DATUM_SHIFT - 1)) + (1 << EXP_SHIFT);
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b:
            return (1 << (DATUM_SHIFT - 2)) + (1 << EXP_SHIFT);
        default:
            return 0;
    }
}

template <DataFormat format>
ALWI constexpr uint32_t get_full_tile_size() {
    return get_full_tile_size_impl(format);
}

ALWI uint32_t get_full_tile_size(DataFormat format) {
    return get_full_tile_size_impl(format);
}

ALWI uint32_t get_cb_num_pages(uint32_t cb_id) {
    auto& cb = get_local_cb_interface(cb_id);
    return cb.fifo_size / cb.fifo_page_size;
}

ALWI constexpr bool is_block_float_format(uint32_t format) {
    return format == 2 || format == 3 || format == 11 || format == 6 || format == 7 || format == 15;
}

template <DataFormat format>
ALWI bool is_valid_cb_tile_page_size(uint32_t cb_id) {
    uint32_t tile_size = get_full_tile_size<format>();
    uint32_t page_size_bytes = get_local_cb_interface(cb_id).fifo_page_size << CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
    return page_size_bytes == tile_size;
}

ALWI bool is_valid_cb_tile_page_size(uint32_t cb_id, DataFormat format) {
    uint32_t tile_size = get_full_tile_size(format);
    uint32_t page_size_bytes = get_local_cb_interface(cb_id).fifo_page_size << CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
    return page_size_bytes == tile_size;
}

}  // namespace compute_kernel_lib
