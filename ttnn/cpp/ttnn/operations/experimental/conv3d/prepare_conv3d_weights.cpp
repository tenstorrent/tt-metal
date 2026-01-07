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
    const ttnn::Shape& original_weight_shape,
    const ttnn::Shape& output_weight_shape,
    uint32_t num_groups,
    DataType output_dtype) {
    auto pad_weight = [&original_weight_shape, &output_weight_shape, num_groups](
                          const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto src_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        auto output_buffer = std::vector<T>(output_weight_shape.volume(), 0);  // 初始全 0

        uint32_t w_total = output_weight_shape[1];        // 总输出通道
        uint32_t h_per_group = original_weight_shape[0];  // Python 对齐后的单组高度
        uint32_t w_per_group = w_total / num_groups;      // 每组负责的输出通道

        for (uint32_t g = 0; g < num_groups; g++) {
            // 源定位：每一组的有效数据在 Python 矩阵中是横向排列的
            uint32_t src_col_start = g * w_per_group;

            // 目标定位：将每一组推向对角线位置 (Height 增加, Width 增加)
            uint32_t dest_row_start = g * h_per_group;
            uint32_t dest_col_start = g * w_per_group;

            for (uint32_t i = 0; i < h_per_group; i++) {
                for (uint32_t j = 0; j < w_per_group; j++) {
                    // 读取：在原始 H_per_group 范围内按列平移
                    uint32_t src_idx = i * w_total + (src_col_start + j);

                    // 写入：在目标 H_out 范围内按块平移
                    uint32_t dest_idx = (dest_row_start + i) * w_total + (dest_col_start + j);

                    if (src_idx < src_buffer.size()) {
                        output_buffer[dest_idx] = src_buffer[src_idx];
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    // 先以线性 Row-Major 创建，确保内存位置一一对应
    const TensorSpec rm_spec(
        output_weight_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    Tensor rm_tensor = convert_tensor<T>(weight, pad_weight, rm_spec);

    // 最后转为 TILE。因为 h_per_group 在 Python 端已经补齐到 32，
    // 所以 h_out (h_per_group * num_groups) 也是 32 的倍数，TILE 转换会非常安全。
    return rm_tensor;
}

/*
Converts convolution weights to grouped layout with padded zeros
This function will take in a weight tensor with shape [out_channels, in_channels // groups, Kd, Kh, Kw] and return a
newly allocated output tensor with shape [out_channels, in_channels, Kd, Kh, Kw] The extra channels in shape[1] will be
padded with 0 - then the entire weight tensor is convolved with the input tensor - equivalent to convolution if the
input tensor was divided into num_groups for each groupped filter
*/
// 核心调用函数：将 Python 处理后的紧凑矩阵转换为硬件所需的对角分组布局
Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype) {
    // 1. 获取 Python 传来的原始形状 [H_per_group, W_total]
    // 此时 H_per_group = kD * kH * kW * C_in_aligned
    // W_total = out_channels
    ttnn::Shape original_shape = conv_weight_tensor.logical_shape();
    uint32_t h_in = original_shape[0];
    uint32_t w_in = original_shape[1];

    // 2. 计算目标形状：高度需要乘以组数，宽度保持不变
    // 这样才能在矩阵乘法中通过高度方向的 0 来隔离不同组的输入通道
    uint32_t h_out = h_in * num_groups;
    uint32_t w_out = w_in;

    ttnn::Shape output_shape({h_out, w_out});

    // 3. 根据数据类型调用对应的模板 helper
    if (conv_weight_tensor.dtype() == DataType::FLOAT32) {
        return conv3d_group_weight_zero_pad_helper<float>(
            conv_weight_tensor, original_shape, output_shape, num_groups, output_dtype);
    } else if (conv_weight_tensor.dtype() == DataType::BFLOAT16) {
        return conv3d_group_weight_zero_pad_helper<bfloat16>(
            conv_weight_tensor, original_shape, output_shape, num_groups, output_dtype);
    } else {
        TT_THROW("Unsupported data type for conv3d group weight padding");
    }
}

static ttnn::Tensor prepare_conv_weights_internal(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& params, MeshDevice* device) {
    ttnn::Tensor weight_tensor_ = weight_tensor;

    if (params.groups > 1) {
        weight_tensor_ = weight_tensor_.cpu();
        log_info(tt::LogTest, "is_cpu_tensor(weight_tensor_): {}", is_cpu_tensor(weight_tensor_));
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
        weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);
    }

    // 在python端传入了row-major，在这要变成TILE
    weight_tensor_ = ttnn::to_layout(weight_tensor_, Layout::TILE);

    return weight_tensor_;
}

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& config, MeshDevice* device) {
    return prepare_conv_weights_internal(weight_tensor, config, device);
}

}  // namespace ttnn::operations::experimental::conv3d
