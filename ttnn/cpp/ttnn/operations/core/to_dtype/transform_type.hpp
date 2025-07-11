// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/bfloat16.hpp"

#include <algorithm>
#include <type_traits>

namespace ttnn::operations::core::detail {
struct bfloat4_b {};
struct bfloat8_b {};

template <typename SrcType>
auto preprocess_tensor(const tt::tt_metal::Tensor& input_tensor) {
    using namespace tt::tt_metal;
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if constexpr (std::is_same_v<SrcType, bfloat8_b>) {
        return input_tensor.host_storage().transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    } else if constexpr (std::is_same_v<SrcType, bfloat4_b>) {
        return input_tensor.host_storage().transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    } else {
        return input_tensor.host_storage();
    }
}

template <typename DstType, typename U>
auto postprocess_vector(std::vector<U>&& data, const tt::tt_metal::Tensor& input_tensor) {
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

    TT_FATAL(is_cpu_tensor(input_tensor), "transform_type(...) function only supports host tensors!");
    auto input_dtype = input_tensor.dtype();
    auto storage = preprocess_tensor<SrcType>(input_tensor);

    using intm_src_t =
        std::conditional_t<std::is_same_v<SrcType, bfloat4_b> || std::is_same_v<SrcType, bfloat8_b>, float, SrcType>;
    using intm_dst_t =
        std::conditional_t<std::is_same_v<DstType, bfloat4_b> || std::is_same_v<DstType, bfloat8_b>, float, DstType>;

    auto output_storage = storage.transform([&](const tt::tt_metal::HostBuffer& host_buffer) {
        auto buffer = host_buffer.view_as<intm_src_t>();
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
        return postprocess_vector<DstType>(std::move(output_vector), input_tensor);
    });

    auto layout =
        std::is_same_v<DstType, bfloat4_b> || std::is_same_v<DstType, bfloat8_b> ? Layout::TILE : input_tensor.layout();

    auto spec = TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dst_type,
            tt::tt_metal::PageConfig(layout),
            MemoryConfig{},
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));

    return Tensor(std::move(output_storage), spec, input_tensor.distributed_tensor_config());
}

}  // namespace ttnn::operations::core::detail
