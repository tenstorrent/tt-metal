// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tuple>

#include <tt-metalium/constants.hpp>

constexpr uint32_t ONE_TILE = 1;
constexpr uint32_t FIRST_TILE = 0;

constexpr uint32_t TILE_FACES_PER_AXIS = 2;
constexpr uint32_t TILE_FACE_HEIGHT = tt::constants::TILE_HEIGHT / TILE_FACES_PER_AXIS;
constexpr uint32_t TILE_FACE_WIDTH = tt::constants::TILE_WIDTH / TILE_FACES_PER_AXIS;
constexpr uint32_t TILE_FACE_HW = TILE_FACE_WIDTH * TILE_FACE_HEIGHT;

template <bool is_dram>
using IAGF = InterleavedAddrGenFast<is_dram>;

// TODO(jbbieniekTT): return immediately calculated result using face bit mask
// (this will speed up the scatter kernel altogether - this method is called
// a multitude of times)
FORCE_INLINE uint32_t calc_offset_inside_tile(
    const std::size_t& face_x, const std::size_t& face_y, const std::size_t& scalar_x, const std::size_t& scalar_y) {
    uint32_t offset = 0;

    // choose the quarter of interest
    if (face_x == 1 && face_y == 0) {
        offset += TILE_FACE_HW * 1;
    } else if (face_x == 0 && face_y == 1) {
        offset += TILE_FACE_HW * 2;
    } else if (face_x == 1 && face_y == 1) {
        offset += TILE_FACE_HW * 3;
    }

    offset += scalar_y * TILE_FACE_WIDTH + scalar_x;

    return offset;
}

FORCE_INLINE uint32_t get_tile_offset_in_row(const uint32_t& tile_id) { return tt::constants::TILE_HW * tile_id; }

template <typename T>
FORCE_INLINE volatile T& tile_guts(
    volatile tt_l1_ptr T* l1_ptr,
    const std::size_t& face_x,
    const std::size_t& face_y,
    const std::size_t& scalar_x,
    const std::size_t& scalar_y,
    const uint32_t& tile_id = 0) {
    return l1_ptr[get_tile_offset_in_row(tile_id) + calc_offset_inside_tile(face_x, face_y, scalar_x, scalar_y)];
}

FORCE_INLINE uint32_t
get_width_scalar_index(const uint32_t& tile_id, const uint32_t& face_x, const uint32_t& scalar_x) {
    return tile_id * tt::constants::TILE_WIDTH + face_x * (tt::constants::TILE_WIDTH / 2) + scalar_x;
}

FORCE_INLINE uint32_t get_height_scalar_index(const uint32_t& h, const uint32_t& face_y, const uint32_t& scalar_y) {
    return h * tt::constants::TILE_HEIGHT + face_y * (tt::constants::TILE_HEIGHT / 2) + scalar_y;
}

template <bool is_dram>
FORCE_INLINE IAGF<is_dram> make_addr_gtor(const uint32_t& cb, const uint32_t& base_addr) {
    return {
        .bank_base_address = base_addr,
        .page_size = static_cast<uint32_t>(get_tile_size(cb)),
        .data_format = get_dataformat(cb)};
}

FORCE_INLINE std::size_t calculate_ht_offset_for_core(
    const uint32_t& Ht, const uint32_t& total_number_of_cores, const uint32_t& compute_with_storage_grid_size_x) {
    const uint32_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    return (core_id * Ht) / total_number_of_cores;
}

struct ScatterCTAs {
    const bool input_tensor_is_dram;
    const bool index_tensor_is_dram;
    const bool source_tensor_is_dram;
    const bool output_tensor_is_dram;
    const uint32_t input_tensor_addr;
    const uint32_t index_tensor_addr;
    const uint32_t source_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t input_tensor_cb;
    const uint32_t index_tensor_cb;
    const uint32_t source_tensor_cb;
    const uint32_t output_tensor_cb;
    const uint32_t Wt_input;
    const uint32_t logical_index_width;
    const uint32_t logical_index_height;
    const uint32_t Wt_index;
    const uint32_t Ht;
    const uint32_t total_number_of_cores;
    const uint32_t compute_with_storage_grid_size_x;
};

constexpr ScatterCTAs get_ctas() {
    return {
        get_compile_time_arg_val(0) == 1,
        get_compile_time_arg_val(1) == 1,
        get_compile_time_arg_val(2) == 1,
        get_compile_time_arg_val(3) == 1,
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(18)};
}
