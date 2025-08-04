// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>

constexpr uint32_t ONE_PAGE = 1;

// choose the right C++ POD type at compile-time
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

struct ScatterCTAs {
    const bool input_tensor_is_dram;
    const bool index_tensor_is_dram;
    const bool source_tensor_is_dram;
    const bool output_tensor_is_dram;
    const uint32_t input_tensor_addr;
    const uint32_t index_tensor_addr;
    const uint32_t source_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t input_cb;
    const uint32_t index_cb;
    const uint32_t source_cb;
    const uint32_t output_cb;
    const uint32_t input_stick_size;
    const uint32_t index_stick_size;
    const uint32_t source_stick_size;
    const uint32_t output_stick_size;
    const uint32_t input_stick_size_bytes;
    const uint32_t index_stick_size_bytes;
    const uint32_t source_stick_size_bytes;
    const uint32_t output_stick_size_bytes;
    const uint32_t input_stick_size_bytes_log2;
    const uint32_t index_stick_size_bytes_log2;
    const uint32_t source_stick_size_bytes_log2;
    const uint32_t output_stick_size_bytes_log2;
    const bool is_input_stick_size_bytes_pow2_min_32;   // necessary for InterleavedAddrGen
    const bool is_index_stick_size_bytes_pow2_min_32;   // necessary for InterleavedAddrGen
    const bool is_source_stick_size_bytes_pow2_min_32;  // necessary for InterleavedAddrGen
    const bool is_output_stick_size_bytes_pow2_min_32;  // necessary for InterleavedAddrGen
};

FORCE_INLINE constexpr ScatterCTAs get_ctas() {
    return {get_compile_time_arg_val(0) == 1,  get_compile_time_arg_val(1) == 1,  get_compile_time_arg_val(2) == 1,
            get_compile_time_arg_val(3) == 1,  get_compile_time_arg_val(4),       get_compile_time_arg_val(5),
            get_compile_time_arg_val(6),       get_compile_time_arg_val(7),       get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),       get_compile_time_arg_val(10),      get_compile_time_arg_val(11),
            get_compile_time_arg_val(12),      get_compile_time_arg_val(13),      get_compile_time_arg_val(14),
            get_compile_time_arg_val(15),      get_compile_time_arg_val(16),      get_compile_time_arg_val(17),
            get_compile_time_arg_val(18),      get_compile_time_arg_val(19),      get_compile_time_arg_val(20),
            get_compile_time_arg_val(21),      get_compile_time_arg_val(22),      get_compile_time_arg_val(23),
            get_compile_time_arg_val(24) == 1, get_compile_time_arg_val(25) == 1, get_compile_time_arg_val(26) == 1,
            get_compile_time_arg_val(27) == 1};
}
