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
    log_info(tt::LogTest, "convert_tensor_to_tiled_layout_common");
    return entry->second(input_tensor, std::forward<Args>(args)..., output_dtype.value_or(input_tensor.dtype()));
}

/*
Helper function to aid in converting grouped weight tensor to ungrouped weight tensor with padded zero channels
*/
template <typename T>
static Tensor conv3d_group_weight_zero_pad_helper(
    const Tensor& weight,                      // 已经是 2D 矩阵 [H_in, W_in]
    const ttnn::Shape& original_weight_shape,  // 这里的 shape 也是 2D [H_in, W_in]
    const ttnn::Shape& output_weight_shape,    // 目标矩阵 2D [H_out, W_out]
    uint32_t num_groups,
    DataType output_dtype) {
    auto pad_weight = [&original_weight_shape, &output_weight_shape, num_groups](
                          const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto src_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        auto output_buffer = std::vector<T>(output_weight_shape.volume(), 0);

        uint32_t h_out = output_weight_shape[0];  // kD * kH * kW * C_in_total
        uint32_t w_out = output_weight_shape[1];  // C_out_total (即 out_channels)

        // 计算每个 Group 在矩阵中占据的“块”大小
        // 每个块的高度包含了该组的所有空间维度和通道数据
        uint32_t rows_per_group = h_out / num_groups;
        uint32_t cols_per_group = w_out / num_groups;

        // 遍历每一个 Group，进行对角块拷贝
        for (uint32_t g = 0; g < num_groups; g++) {
            // 目标矩阵中的起始坐标 (对角块位置)
            uint32_t dest_start_row = g * rows_per_group;
            uint32_t dest_start_col = g * cols_per_group;

            // 源矩阵中的起始坐标
            // 假设源矩阵是 [kD*kH*kW * (C_in/G) * G, (C_out/G)] 这种紧凑排布
            uint32_t src_start_row = g * rows_per_group;

            for (uint32_t i = 0; i < rows_per_group; i++) {
                for (uint32_t j = 0; j < cols_per_group; j++) {
                    // 计算源矩阵的一维索引 (源矩阵宽度为 cols_per_group)
                    uint32_t src_row = src_start_row + i;
                    uint32_t src_col = j;
                    uint32_t src_idx = src_row * cols_per_group + src_col;

                    // 计算目标矩阵的一维索引 (目标矩阵宽度为 w_out)
                    uint32_t dest_row = dest_start_row + i;
                    uint32_t dest_col = dest_start_col + j;
                    uint32_t dest_idx = dest_row * w_out + dest_col;

                    // 赋值
                    if (src_idx < src_buffer.size()) {
                        output_buffer[dest_idx] = src_buffer[src_idx];
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    const TensorSpec output_spec(
        output_weight_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));

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
    const auto& original_shape = conv_weight_tensor.logical_shape();
    uint32_t h_in = original_shape[0];
    uint32_t w_in = original_shape[1];  // 这通常是 out_channels / num_groups

    // 计算输出矩阵的形状：宽度变为原来的 num_groups 倍（补 0 后的总 Out Channels）
    // 高度保持一致，因为 H_in 已经包含了 kD*kH*kW*C_in_per_group * num_groups
    uint32_t h_out = h_in;
    uint32_t w_out = w_in * num_groups;

    ttnn::Shape output_conv_weight_tensor_shape(ttnn::SmallVector<uint32_t>{h_out, w_out});

    const static std::
        unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::Shape, ttnn::Shape, uint32_t, DataType)>>
            to_w_tile_layout_map = {
                {DataType::INT32, &conv3d_group_weight_zero_pad_helper<int32_t>},
                {DataType::FLOAT32, &conv3d_group_weight_zero_pad_helper<float>},
                {DataType::BFLOAT16, &conv3d_group_weight_zero_pad_helper<bfloat16>},
                {DataType::UINT16, &conv3d_group_weight_zero_pad_helper<uint16_t>},
                {DataType::BFLOAT8_B, &conv3d_group_weight_zero_pad_helper<float>},
                {DataType::UINT32, &conv3d_group_weight_zero_pad_helper<uint32_t>},
                {DataType::BFLOAT4_B, &conv3d_group_weight_zero_pad_helper<uint32_t>},
            };

    if (tt::tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for Conv3D with groups > 1 expects weights on host, but they are on device. The op will "
            "move them back to host.");
    }

    log_info(tt::LogTest, "conv_weight_tensor.layout(): {}", conv_weight_tensor.layout());
    Tensor weight_on_host = tt::tt_metal::is_device_tensor(conv_weight_tensor)
                                ? ttnn::operations::core::from_device(conv_weight_tensor)
                                : conv_weight_tensor;
    log_info(tt::LogTest, "weight_on_host.logical_shape(): {}", weight_on_host.logical_shape());
    log_info(tt::LogTest, "weight_on_host.layout(): {}", weight_on_host.layout());

    return convert_tensor_to_tiled_layout_common(
        weight_on_host,
        output_dtype,
        to_w_tile_layout_map,
        original_shape,
        output_conv_weight_tensor_shape,
        num_groups);
}

static ttnn::Tensor prepare_conv_weights_internal(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& params, MeshDevice* device) {
    ttnn::Tensor weight_tensor_ = weight_tensor;

    if (params.groups > 1) {
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());

    }
    // group=1,需要实现原先python端中的pad与tile操作
    else {
        weight_tensor_ = ttnn::to_layout(weight_tensor_, Layout::TILE);
    }
    log_info(tt::LogTest, "weight_tensor_.logical_shape(): {}", weight_tensor_.logical_shape());
    log_info(tt::LogTest, "weight_tensor_.layout(): {}", weight_tensor_.layout());

    log_info(tt::LogTest, "ttnn::operations::core::to_device");
    // Always move parameters to device
    weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);

    log_info(tt::LogTest, "ttnn::operations::core::to_device END END END");

    return weight_tensor_;
}

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& config, MeshDevice* device) {
    return prepare_conv_weights_internal(weight_tensor, config, device);
}

}  // namespace ttnn::operations::experimental::conv3d
