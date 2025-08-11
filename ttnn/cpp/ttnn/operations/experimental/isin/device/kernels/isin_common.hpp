// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

using index_hint_number_type = uint32_t;
using output_number_type = uint8_t;

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

constexpr uint32_t ONE_PAGE = 1;
constexpr uint32_t FIRST_STICK = 0;

template <
    typename elements_accessor_args_type,
    typename test_elements_accessor_args_type,
    typename output_accessor_args_type>
struct IsInCTAs {
    const uint32_t elements_tensor_addr;
    const uint32_t test_elements_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t elements_cb;
    const uint32_t test_elements_cb;
    const uint32_t output_cb;
    const uint32_t elements_size;
    const uint32_t test_elements_size;
    const uint32_t single_fetch_subchunk_size;
    const bool invert;
    const uint32_t num_cores;
    const elements_accessor_args_type elements_accessor_args;
    const test_elements_accessor_args_type test_elements_accessor_args;
    const output_accessor_args_type output_accessor_args;
};

FORCE_INLINE constexpr IsInCTAs get_ctas() {
    constexpr auto elements_args = TensorAccessorArgs<11>{};
    constexpr auto test_elements_args = TensorAccessorArgs<elements_args.next_compile_time_args_offset()>{};
    constexpr auto output_args = TensorAccessorArgs<test_elements_args.next_compile_time_args_offset()>{};
    return {
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10) != 0,
        get_compile_time_arg_val(11),
        elements_args,
        test_elements_args,
        output_args};
}
