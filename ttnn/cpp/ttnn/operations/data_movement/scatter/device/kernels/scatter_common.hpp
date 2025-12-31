// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.hpp"

#include <array>
#include <cstdint>

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
