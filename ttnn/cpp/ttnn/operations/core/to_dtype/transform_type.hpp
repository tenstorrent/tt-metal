// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/bfloat16.hpp"

#include <algorithm>
#include <type_traits>

namespace ttnn::operations::core::detail {
struct bfloat4_b {};
struct bfloat8_b {};

template <typename SrcType>
auto preprocess_buffer(const tt::tt_metal::HostBuffer& buffer) {
    using namespace tt::tt_metal;
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if constexpr (std::is_same_v<SrcType, bfloat8_b>) {
        tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
        return unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
    } else if constexpr (std::is_same_v<SrcType, bfloat4_b>) {
        tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
        return unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
    } else {
        return buffer.view_as<SrcType>();
    }
}

template <typename DstType, typename U>
auto postprocess_buffer(std::vector<U>&& data, const tt::tt_metal::Tensor& input_tensor) {
    using namespace tt::tt_metal;
    constexpr bool row_major_input = false;
    constexpr bool is_exp_a = false;

    if constexpr (std::is_same_v<DstType, bfloat8_b> || std::is_same_v<DstType, bfloat4_b>) {
        if (input_tensor.layout() == Layout::ROW_MAJOR) {
            data = tensor_impl::convert_layout_row_major_to_tile(
                input_tensor.tensor_spec().physical_shape(),
                input_tensor.tensor_spec().tile(),
                tt::stl::make_const_span(data));
        }
        return tt::tt_metal::HostBuffer(
            std::is_same_v<DstType, bfloat8_b> ? pack_fp32_vec_as_bfp8_tiles(data, row_major_input, is_exp_a)
                                               : pack_fp32_vec_as_bfp4_tiles(data, row_major_input, is_exp_a));
    }
    return tt::tt_metal::HostBuffer(std::move(data));
}

template <typename SrcType, typename DstType>
tt::tt_metal::Tensor transform_type(const tt::tt_metal::Tensor& input_tensor, const DataType dst_type) {
    using namespace tt::tt_metal;

    auto input_dtype = input_tensor.dtype();
    auto buffer = preprocess_buffer<SrcType>(std::get<HostStorage>(input_tensor.storage()).buffer);

    using intm_dst_t =
        std::conditional_t<std::is_same_v<DstType, bfloat4_b> || std::is_same_v<DstType, bfloat8_b>, float, DstType>;

    std::vector<intm_dst_t> output_vector(buffer.size());
    std::transform(buffer.begin(), buffer.end(), output_vector.begin(), [](const auto& value) {
        if constexpr (std::is_same_v<SrcType, ::bfloat16>) {
            return intm_dst_t(value.to_float());
        } else if constexpr (std::is_same_v<intm_dst_t, ::bfloat16>) {
            return intm_dst_t(static_cast<float>(value));
        } else {
            return intm_dst_t(value);
        }
    });

    auto output_buffer = postprocess_buffer<DstType>(std::move(output_vector), input_tensor);
    return Tensor(
        std::move(output_buffer),
        input_tensor.logical_shape(),
        input_tensor.padded_shape(),
        dst_type,
        dst_type == tt::tt_metal::DataType::BFLOAT4_B or dst_type == tt::tt_metal::DataType::BFLOAT8_B
            ? Layout::TILE
            : input_tensor.layout());
}

}  // namespace ttnn::operations::core::detail
