// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_conv3d_weights.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
// #include "ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include <vector>
#include <algorithm>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

// Helper functions copied/adapted from conv2d

template <typename T, typename Fn>
Tensor convert_tensor(const Tensor& input_tensor, const Fn& compute, const TensorSpec& output_spec) {
    TT_FATAL(is_cpu_tensor(input_tensor), "convert_tensor only supports cpu tensors");
    return Tensor(input_tensor.host_storage().transform(compute), output_spec, input_tensor.tensor_topology());
}

template <typename Func, typename... Args>
Tensor convert_tensor_to_tiled_layout_common(
    const Tensor& input_tensor,
    std::optional<DataType> output_dtype,
    const std::unordered_map<DataType, Func>& function_map,
    Args&&... args) {
    TT_ASSERT(
        input_tensor.layout() == Layout::ROW_MAJOR &&
        "Tensor(weight/bias) should be in row major layout for conversion to tilized layout.");

    auto entry = function_map.find(input_tensor.dtype());
    if (entry == function_map.end()) {
        TT_THROW("Unsupported data type");
    }
    return entry->second(input_tensor, std::forward<Args>(args)..., output_dtype.value_or(input_tensor.dtype()));
}

/*
Helper function to aid in converting grouped weight tensor to ungrouped weight tensor with padded zero channels
*/
template <typename T>
static Tensor conv3d_group_weight_zero_pad_helper(
    const Tensor& weight,
    const ttnn::Shape& original_2d_shape,  // [H_total, oC_pg]
    const ttnn::Shape& output_2d_shape,    // [H_total, oC_total]
    uint32_t num_groups,
    // uint32_t iC_per_group_block,           // 必须等于 Python 中的 C_in_block (如 32)
    DataType output_dtype) {
    uint32_t iC_per_group_block = 32;

    auto pad_weight = [&original_2d_shape, &output_2d_shape, num_groups, iC_per_group_block](
                          const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto src_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        auto output_buffer = std::vector<T>(output_2d_shape.volume(), 0);

        uint32_t h_total = original_2d_shape[0];
        uint32_t w_per_group = original_2d_shape[1];  // oC_pg
        uint32_t w_total = output_2d_shape[1];        // oC_total

        for (uint32_t i = 0; i < h_total; ++i) {
            uint32_t group_id = (i / iC_per_group_block) % num_groups;

            uint32_t dest_col_start = group_id * w_per_group;

            for (uint32_t j = 0; j < w_per_group; j++) {
                uint64_t src_idx = (uint64_t)i * w_per_group + j;
                uint64_t dest_idx = (uint64_t)i * w_total + (dest_col_start + j);

                if (src_idx < src_buffer.size()) {
                    output_buffer[dest_idx] = src_buffer[src_idx];
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    const TensorSpec rm_spec(
        output_2d_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return convert_tensor<T>(weight, pad_weight, rm_spec);
}

/*
Converts convolution weights to grouped layout with padded zeros
This function will take in a weight tensor with shape [out_channels, in_channels // groups, Kd, Kh, Kw] and return a
newly allocated output tensor with shape [out_channels, in_channels, Kd, Kh, Kw] The extra channels in shape[1] will be
padded with 0 - then the entire weight tensor is convolved with the input tensor - equivalent to convolution if the
input tensor was divided into num_groups for each groupped filter
*/
Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype) {
    ttnn::Shape original_shape = conv_weight_tensor.logical_shape();
    uint32_t h_in = original_shape[0];
    uint32_t w_in = original_shape[1];

    uint32_t h_out = h_in;
    uint32_t w_out = w_in * num_groups;

    ttnn::Shape output_shape({h_out, w_out});

    const static std::unordered_map<
        DataType,
        std::function<Tensor(const Tensor&, const ttnn::Shape&, const ttnn::Shape&, uint32_t, DataType)>>
        to_w_tile_layout_map = {
            {DataType::INT32, &conv3d_group_weight_zero_pad_helper<int32_t>},
            {DataType::FLOAT32, &conv3d_group_weight_zero_pad_helper<float>},
            {DataType::BFLOAT16, &conv3d_group_weight_zero_pad_helper<bfloat16>},
            {DataType::UINT16, &conv3d_group_weight_zero_pad_helper<uint16_t>},
            {DataType::BFLOAT8_B, &conv3d_group_weight_zero_pad_helper<float>},
            {DataType::UINT32, &conv3d_group_weight_zero_pad_helper<uint32_t>},
            {DataType::BFLOAT4_B, &conv3d_group_weight_zero_pad_helper<uint32_t>},
        };

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor, output_dtype, to_w_tile_layout_map, original_shape, output_shape, num_groups);
}

static ttnn::Tensor prepare_conv_weights_internal(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& params, MeshDevice* device) {
    ttnn::Tensor weight_tensor_ = weight_tensor;

    if (params.groups > 0) {
        weight_tensor_ = weight_tensor_.cpu();
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
        weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);
    }
    weight_tensor_ = ttnn::to_layout(weight_tensor_, Layout::TILE);

    return weight_tensor_;
}

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& config, MeshDevice* device) {
    return prepare_conv_weights_internal(weight_tensor, config, device);
}

}  // namespace ttnn::operations::experimental::conv3d
