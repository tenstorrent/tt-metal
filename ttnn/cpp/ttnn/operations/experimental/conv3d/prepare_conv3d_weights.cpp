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
static Tensor conv_group_weight_zero_pad_helper(
    const Tensor& weight,
    const ttnn::Shape& original_weight_shape,
    const ttnn::Shape& output_weight_shape,
    uint32_t num_groups,
    DataType output_dtype) {
    auto pad_weight = [&original_weight_shape, &output_weight_shape, num_groups, output_dtype](
                          const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto conv_weight_tensor_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        auto output_buffer = std::vector<T>(output_weight_shape.volume());

        for (int curr_batch_idx = 0; curr_batch_idx < original_weight_shape[0]; curr_batch_idx++) {
            int new_batch_idx = curr_batch_idx;

            // Find which group_id the filter belongs to - through this, we can compute the offset where the padding
            // should be applied
            auto group_size = original_weight_shape[0] / num_groups;
            auto group_index = curr_batch_idx / group_size;
            auto group_id = std::min(group_index, num_groups - 1);
            int new_channel_start_idx = group_id * original_weight_shape[1];

            for (int j = 0; j < original_weight_shape[1]; j++) {
                for (int k = 0; k < original_weight_shape[2]; k++) {
                    for (int l = 0; l < original_weight_shape[3]; l++) {
                        for (int m = 0; m < original_weight_shape[4]; m++) {
                            // Get value from original weight tensor
                            auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                                ttnn::SmallVector<int>{curr_batch_idx, j, k, l, m},
                                compute_strides(original_weight_shape));
                            auto value = conv_weight_tensor_buffer[value_flat_input_index];

                            // Copy value to output tensor at the adjusted position
                            auto new_channel_idx = new_channel_start_idx + j;
                            auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                                ttnn::SmallVector<int>{new_batch_idx, new_channel_idx, k, l, m},
                                compute_strides(output_weight_shape));
                            output_buffer[output_flat_input_index] = value;
                        }
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    const TensorSpec output_spec(
        output_weight_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return convert_tensor<T>(weight, pad_weight, output_spec);
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
    // Define output tensor shape. This is going to be channel dimension of weight tensor * num_groups - this value
    // should match number of input channels being convolved with the weight tensor
    const auto& original_conv_weight_tensor_shape = conv_weight_tensor.logical_shape();
    ttnn::Shape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        original_conv_weight_tensor_shape[1] * num_groups,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3],
        original_conv_weight_tensor_shape[4]};

    const static std::
        unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::Shape, ttnn::Shape, uint32_t, DataType)>>
            to_w_tile_layout_map = {
                {DataType::INT32, &conv_group_weight_zero_pad_helper<int32_t>},
                {DataType::FLOAT32, &conv_group_weight_zero_pad_helper<float>},
                {DataType::BFLOAT16, &conv_group_weight_zero_pad_helper<bfloat16>},
                {DataType::UINT16, &conv_group_weight_zero_pad_helper<uint16_t>},
                {DataType::BFLOAT8_B, &conv_group_weight_zero_pad_helper<float>},
                {DataType::UINT32, &conv_group_weight_zero_pad_helper<uint32_t>},
                {DataType::BFLOAT4_B, &conv_group_weight_zero_pad_helper<uint32_t>},
            };

    if (tt::tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for Conv3D with groups > 1 expects weights on host, but they are on device. The op will "
            "move them back to host.");
    }
    return convert_tensor_to_tiled_layout_common(
        tt::tt_metal::is_device_tensor(conv_weight_tensor) ? ttnn::operations::core::from_device(conv_weight_tensor)
                                                           : conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape,
        num_groups);
}

static ttnn::Tensor prepare_conv_weights_internal(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& params) {
    ttnn::Tensor weight_tensor_ = weight_tensor;

    if (params.groups > 1) {
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
    }

    return weight_tensor_;
}

ttnn::Tensor prepare_conv_weights(const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& config) {
    return prepare_conv_weights_internal(weight_tensor, config);
}

}  // namespace ttnn::operations::experimental::conv3d
