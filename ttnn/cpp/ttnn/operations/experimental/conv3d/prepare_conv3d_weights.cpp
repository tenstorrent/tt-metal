// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_conv3d_weights.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include <vector>
#include <algorithm>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

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

        auto original_strides = compute_strides(original_weight_shape);
        auto output_strides = compute_strides(output_weight_shape);

        for (int curr_batch_idx = 0; curr_batch_idx < original_weight_shape[0]; curr_batch_idx++) {
            int new_batch_idx = curr_batch_idx;

            auto group_size = original_weight_shape[0] / num_groups;
            auto group_index = curr_batch_idx / group_size;
            auto group_id = std::min(group_index, num_groups - 1);
            int new_channel_start_idx = group_id * original_weight_shape[1];

            for (int j = 0; j < original_weight_shape[1]; j++) {
                for (int d = 0; d < original_weight_shape[2]; d++) {
                    for (int k = 0; k < original_weight_shape[3]; k++) {
                        for (int m = 0; m < original_weight_shape[4]; m++) {
                            auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                                ttnn::SmallVector<uint32_t>{
                                    (uint32_t)curr_batch_idx, (uint32_t)j, (uint32_t)d, (uint32_t)k, (uint32_t)m},
                                original_strides);
                            auto value = conv_weight_tensor_buffer[value_flat_input_index];

                            auto new_channel_idx = new_channel_start_idx + j;
                            auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                                ttnn::SmallVector<uint32_t>{
                                    (uint32_t)new_batch_idx,
                                    (uint32_t)new_channel_idx,
                                    (uint32_t)d,
                                    (uint32_t)k,
                                    (uint32_t)m},
                                output_strides);
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

Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype) {
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

Tensor prepare_weights(
    const ttnn::Tensor& weights, uint32_t groups, uint32_t C_in_block, MeshDevice* device, uint32_t alignment) {
    ttnn::Tensor prepare_weights = weights;

    prepare_weights = convert_conv_weight_tensor_to_grouped_layout(weights, groups, weights.dtype());
    prepare_weights = ttnn::operations::core::to_device(prepare_weights, device, std::nullopt);

    ttnn::SmallVector<int64_t> dims_1 = {2, 3, 4, 1, 0};
    prepare_weights = ttnn::permute(prepare_weights, dims_1);
    uint32_t C = prepare_weights.logical_shape()[3];
    uint32_t ALIGN_PAD = alignment - (C % alignment);
    if (C % alignment != 0) {
        ttnn::SmallVector<std::array<uint32_t, 2>> padding_shape({{0, 0}, {0, 0}, {0, 0}, {0, ALIGN_PAD}, {0, 0}});
        prepare_weights = ttnn::pad(prepare_weights, padding_shape, 0.0f);
    }
    // Reshape and permute weights
    auto weights_shape = prepare_weights.logical_shape();
    auto kD = weights_shape[0];
    auto kH = weights_shape[1];
    auto kW = weights_shape[2];
    auto C_in_aligned = weights_shape[3];
    auto out_channels = weights_shape[4];

    if (C_in_block == 0) {
        C_in_block = C_in_aligned;
    }
    uint32_t num_C_in_blocks = C_in_aligned / C_in_block;
    TT_FATAL(num_C_in_blocks * C_in_block == C_in_aligned, "C_in_aligned must be divisible by C_in_block");

    prepare_weights =
        ttnn::reshape(prepare_weights, ttnn::Shape{kD, kH, kW, num_C_in_blocks, C_in_block, out_channels});
    ttnn::SmallVector<int64_t> dims_2 = {3, 0, 1, 2, 4, 5};
    prepare_weights = ttnn::permute(prepare_weights, dims_2);
    prepare_weights = ttnn::reshape(prepare_weights, ttnn::Shape{-1, out_channels});
    return prepare_weights;
}

}  // namespace ttnn::operations::experimental::conv3d
