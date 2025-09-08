// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>

#include <array>
#include <cstdint>

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

template <
    typename InputAccessorArgs,
    typename IndexAccessorArgs,
    typename SourceAccessorArgs,
    typename OutputAccessorArgs>
struct ScatterCTAs {
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
    const uint32_t input_rank;
    const InputAccessorArgs input_args;
    const IndexAccessorArgs index_args;
    const SourceAccessorArgs source_args;
    const OutputAccessorArgs output_args;
};

FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto input_args = TensorAccessorArgs<17>();
    constexpr auto index_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto source_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    return ScatterCTAs<decltype(input_args), decltype(index_args), decltype(source_args), decltype(output_args)>{
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
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        input_args,
        index_args,
        source_args,
        output_args,
    };
}

template <uint32_t N>
std::array<uint32_t, N> make_shape_array_from_runtime_args(const uint32_t& C) {
    std::array<uint32_t, N> ret{};
    for (uint32_t i = C; i < C + N; ++i) {
        ret[i - C] = get_arg_val<uint32_t>(i);
    }

    return ret;
}
