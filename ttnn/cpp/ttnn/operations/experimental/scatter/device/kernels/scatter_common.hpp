// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>

constexpr uint32_t ONE_TILE = 1;
constexpr uint32_t FIRST_TILE = 0;

template <bool is_dram>
using IAGF = InterleavedAddrGenFast<is_dram>;

template <DataFormat df>
struct df_to_std {
    using std_type = void;
};

template <>
struct df_to_std<DataFormat::Float32> {
    using std_type = float;
};

template <>
struct df_to_std<DataFormat::Float16_b> {
    using std_type = uint16_t;
};

template <>
struct df_to_std<DataFormat::Int32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::UInt32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::UInt16> {
    using std_type = uint16_t;
};

template <>
struct df_to_std<DataFormat::UInt8> {
    using std_type = uint8_t;
};

template <DataFormat df>
using std_type_t = typename df_to_std<df>::std_type;

FORCE_INLINE uint32_t calc_offset_inside_tile(
    const uint32_t& face_x,
    const uint32_t& face_y,
    const uint32_t& scalar_x,
    const uint32_t& scalar_y,
    const uint32_t& face_hw,
    const uint32_t& face_width) {
    // pick the face
    const uint32_t face_multiplier = ((face_y << 1) | (face_x));
    uint32_t offset = face_hw * face_multiplier;

    // pick the value inside face
    offset += scalar_y * face_width + scalar_x;

    return offset;
}

template <typename T>
FORCE_INLINE volatile T& tile_guts(
    volatile tt_l1_ptr T* l1_ptr,
    const uint32_t& face_x,
    const uint32_t& face_y,
    const uint32_t& scalar_x,
    const uint32_t& scalar_y,
    const uint32_t& face_hw,
    const uint32_t& face_width) {
    return l1_ptr[calc_offset_inside_tile(face_x, face_y, scalar_x, scalar_y, face_hw, face_width)];
}

FORCE_INLINE uint32_t get_width_scalar_index(
    const uint32_t& tile_id,
    const uint32_t& face_x,
    const uint32_t& scalar_x,
    const uint32_t& tile_width,
    const uint32_t& face_width) {
    return tile_id * tile_width + face_x * face_width + scalar_x;
}

FORCE_INLINE uint32_t
get_height_scalar_index_inside_tile(const uint32_t& face_y, const uint32_t& scalar_y, const uint32_t& face_height) {
    return face_y * face_height + scalar_y;
}

template <bool is_dram>
FORCE_INLINE IAGF<is_dram> make_addr_gtor(const uint32_t& cb, const uint32_t& base_addr) {
    return {
        .bank_base_address = base_addr,
        .page_size = static_cast<uint32_t>(get_tile_size(cb)),
        .data_format = get_dataformat(cb)};
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
    const uint32_t face_num_x;
    const uint32_t face_num_y;
    const uint32_t Wt_index;
    const uint32_t pad_scalar_offset;
    const uint32_t Ht;
    const uint32_t total_number_of_cores;
    const uint32_t compute_with_storage_grid_size_x;
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t face_height;
    const uint32_t face_width;
};

FORCE_INLINE constexpr ScatterCTAs get_ctas() {
    return {get_compile_time_arg_val(0) == 1, get_compile_time_arg_val(1) == 1, get_compile_time_arg_val(2) == 1,
            get_compile_time_arg_val(3) == 1, get_compile_time_arg_val(4),      get_compile_time_arg_val(5),
            get_compile_time_arg_val(6),      get_compile_time_arg_val(7),      get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),      get_compile_time_arg_val(10),     get_compile_time_arg_val(11),
            get_compile_time_arg_val(12),     get_compile_time_arg_val(13),     get_compile_time_arg_val(14),
            get_compile_time_arg_val(15),     get_compile_time_arg_val(16),     get_compile_time_arg_val(17),
            get_compile_time_arg_val(18),     get_compile_time_arg_val(19),     get_compile_time_arg_val(20),
            get_compile_time_arg_val(21),     get_compile_time_arg_val(22),     get_compile_time_arg_val(23),
            get_compile_time_arg_val(24),     get_compile_time_arg_val(25)};
}
