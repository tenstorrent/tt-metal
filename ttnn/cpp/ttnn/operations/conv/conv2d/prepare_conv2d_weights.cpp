// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "tt-metalium/assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include <optional>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
namespace ttnn {
namespace operations::conv {
using namespace tt;
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv2d {

template <typename T, typename compute_>
Tensor convert_tensor(const Tensor& input_tensor, compute_& compute) {
    auto convert_tensor = [&compute](const auto& input_tensor) {
        return std::visit(
            [&compute](auto&& storage) -> Tensor {
                using StorageType = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<StorageType, tt::tt_metal::HostStorage>) {
                    return compute(tt::tt_metal::host_buffer::get_as<T>(storage.buffer));
                } else {
                    TT_THROW("Unsupported storage type");
                }
            },
            input_tensor.storage());
    };

    TT_FATAL(!is_device_tensor(input_tensor), "convert_tensor only supports host tensors");

    // TODO: #15840 - Treat multi-device host vs owned/borrowed tensors uniformly.
    return is_multi_device_host_tensor(input_tensor) ? transform(input_tensor, convert_tensor)
                                                     : convert_tensor(input_tensor);
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
Tensor create_tensor_from_owned_buffer(
    tt::tt_metal::HostBuffer buf, DataType& output_dtype, ttnn::Shape& output_shape) {
    if constexpr (std::is_same<T, float>::value) {
        if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
            auto tensor =
                Tensor(std::move(buf), output_shape, DataType::FLOAT32, Layout::ROW_MAJOR).to_layout(Layout::TILE);
            auto output_float_data = tt::tt_metal::host_buffer::get_as<float>(tensor);
            auto output_packed_data =
                output_dtype == DataType::BFLOAT8_B
                    ? pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false)
                    : pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto output_uint32_buffer = tt::tt_metal::HostBuffer(std::move(output_packed_data));
            return Tensor(std::move(output_uint32_buffer), output_shape, output_dtype, Layout::TILE);
        }
    } else {
        TT_FATAL(
            (output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B),
            "Unsupported output datatype");
    }
    auto rm_tensor = Tensor(std::move(buf), output_shape, output_dtype, Layout::ROW_MAJOR);
    return rm_tensor.to_layout(Layout::TILE);
}

template <typename T>
Tensor to_weight_special_padding_tile_layout(
    const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.padded_shape();
    auto compute = [&w_shape, &in1_block_h, &in1_block_w, &output_dtype](const auto& input_buffer) {
        uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
        uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
        auto weight_matrix_cols = w_shape[0];
        // width padding
        if (weight_matrix_cols % in1_block_w_datums != 0) {
            weight_matrix_cols =
                (uint32_t)std::ceil((double)weight_matrix_cols / (double)in1_block_w_datums) * in1_block_w_datums;
        }
        // height padding
        assert(in1_block_h_datums >= w_shape[1] * w_shape[3]);
        uint32_t block_height_padding = in1_block_h_datums - (w_shape[1] * w_shape[3]);
        auto weight_matrix_rows = ((w_shape[1] * w_shape[3]) + block_height_padding) * w_shape[2];
        ttnn::Shape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = std::vector<T>(output_shape.volume());
        for (auto r = 0; r < w_shape[2]; r++) {
            for (auto s = 0; s < w_shape[3]; s++) {
                for (auto c = 0; c < w_shape[1]; c++) {
                    for (auto k = 0; k < w_shape[0]; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols +
                                          r * ((w_shape[3] * w_shape[1]) + block_height_padding) * weight_matrix_cols;
                        auto idx =
                            k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
                        output_buffer[matrix_idx] = input_buffer[idx];
                    }
                }
            }
        }
        return create_tensor_from_owned_buffer<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };
    return convert_tensor<T>(conv_weight_tensor, compute);
}

template <typename T>
Tensor to_weight_tile_layout(
    const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.padded_shape();
    auto compute = [&w_shape, &in1_block_h, &in1_block_w, &output_dtype](const auto& input_buffer) {
        auto weight_matrix_cols = w_shape[0];
        // width padding
        uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
        if (weight_matrix_cols % in1_block_w_datums != 0) {
            weight_matrix_cols =
                (uint32_t)std::ceil((double)weight_matrix_cols / (double)in1_block_w_datums) * in1_block_w_datums;
        }
        // height padding
        auto weight_matrix_rows = w_shape[1] * w_shape[2] * w_shape[3];
        uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
        if (weight_matrix_rows % in1_block_h_datums != 0) {
            weight_matrix_rows =
                (uint32_t)std::ceil((double)weight_matrix_rows / (double)in1_block_h_datums) * in1_block_h_datums;
        }
        ttnn::Shape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = std::vector<T>(output_shape.volume());
        for (auto r = 0; r < w_shape[2]; r++) {
            for (auto s = 0; s < w_shape[3]; s++) {
                for (auto c = 0; c < w_shape[1]; c++) {
                    for (auto k = 0; k < w_shape[0]; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols +
                                          r * w_shape[3] * w_shape[1] * weight_matrix_cols;
                        auto idx =
                            k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
                        output_buffer[matrix_idx] = input_buffer[idx];
                    }
                }
            }
        }
        return create_tensor_from_owned_buffer<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };

    return convert_tensor<T>(conv_weight_tensor, compute);
}

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, uint32_t, uint32_t, DataType)>>
        to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout<float>},
            {DataType::UINT32, &to_weight_tile_layout<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor, output_dtype, to_w_tile_layout_map, in1_block_h, in1_block_w);
}

template <typename T>
Tensor to_weight_tile_layout_block_sharded(
    const Tensor& conv_weight_tensor, uint32_t num_channel_shards, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.padded_shape();
    auto compute = [&w_shape, &num_channel_shards, &output_dtype](const auto& input_buffer) {
        auto weight_matrix_cols = w_shape[0];
        TT_ASSERT(weight_matrix_cols % num_channel_shards == 0);
        auto conv_output_shard_width = weight_matrix_cols / num_channel_shards;
        auto conv_output_shard_width_padded =
            (uint32_t)std::ceil((double)conv_output_shard_width / (double)constants::TILE_WIDTH) *
            constants::TILE_WIDTH;
        if (conv_output_shard_width < conv_output_shard_width_padded) {
            // width padding for conv output shard padding
            weight_matrix_cols = conv_output_shard_width_padded * num_channel_shards;
        }
        auto weight_matrix_rows = w_shape[1] * w_shape[2] * w_shape[3];
        TT_ASSERT(w_shape[1] % num_channel_shards == 0);
        auto conv_input_shard_width = w_shape[1] / num_channel_shards;
        auto weight_block_height = conv_input_shard_width * w_shape[2] * w_shape[3];
        auto weight_block_height_padded =
            (uint32_t)std::ceil((double)weight_block_height / (double)constants::TILE_HEIGHT) * constants::TILE_HEIGHT;
        if (weight_block_height < weight_block_height_padded) {
            // height padding for non tile multiple block height
            weight_matrix_rows = weight_block_height_padded * num_channel_shards;
        }
        ttnn::Shape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = std::vector<T>(output_shape.volume());
        for (auto ic = 0; ic < num_channel_shards; ic++) {
            for (auto r = 0; r < w_shape[2]; r++) {
                for (auto s = 0; s < w_shape[3]; s++) {
                    for (auto c_s = 0; c_s < conv_input_shard_width; c_s++) {
                        for (auto oc = 0; oc < num_channel_shards; oc++) {
                            for (auto k_s = 0; k_s < conv_output_shard_width; k_s++) {
                                auto matrix_idx = (oc * conv_output_shard_width_padded + k_s) +
                                                  c_s * weight_matrix_cols +
                                                  s * conv_input_shard_width * weight_matrix_cols +
                                                  r * w_shape[3] * conv_input_shard_width * weight_matrix_cols +
                                                  ic * weight_block_height_padded * weight_matrix_cols;
                                auto idx = (oc * conv_output_shard_width + k_s) * w_shape[1] * w_shape[2] * w_shape[3] +
                                           (ic * conv_input_shard_width + c_s) * w_shape[2] * w_shape[3] +
                                           r * w_shape[3] + s;
                                output_buffer[matrix_idx] = input_buffer[idx];
                            }
                        }
                    }
                }
            }
        }
        return create_tensor_from_owned_buffer<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };
    return convert_tensor<T>(conv_weight_tensor, compute);
}

// Converts convolution weights to tilized 2d matrix layout for block sharded conv.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout_block_sharded(
    const Tensor& conv_weight_tensor, uint32_t num_channel_shards, std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, uint32_t, DataType)>>
        to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout_block_sharded<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout_block_sharded<float>},
            {DataType::UINT32, &to_weight_tile_layout_block_sharded<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor, output_dtype, to_w_tile_layout_map, num_channel_shards);
}

template <typename T>
Tensor to_bias_tile_layout_block_sharded(
    const Tensor& conv_bias_tensor, uint32_t num_channel_shards, DataType output_dtype) {
    auto b_shape = conv_bias_tensor.padded_shape();
    TT_ASSERT(b_shape[0] == 1 && b_shape[1] == 1 && b_shape[2] == 1);
    auto compute = [&b_shape, &num_channel_shards, &output_dtype](const auto& input_buffer) {
        auto bias_matrix_cols = b_shape[3];
        /*TT_ASSERT(bias_matrix_cols % num_channel_shards == 0);*/
        auto conv_output_shard_width = bias_matrix_cols / num_channel_shards;
        auto conv_output_shard_width_padded =
            (uint32_t)std::ceil((double)conv_output_shard_width / (double)constants::TILE_WIDTH) *
            constants::TILE_WIDTH;
        if (conv_output_shard_width < conv_output_shard_width_padded) {
            // width padding for conv output shard padding
            bias_matrix_cols = conv_output_shard_width_padded * num_channel_shards;
        }

        auto bias_matrix_rows = 32;
        ttnn::Shape output_shape{1, 1, bias_matrix_rows, bias_matrix_cols};
        auto output_buffer = std::vector<T>(output_shape.volume());
        for (auto oc = 0; oc < num_channel_shards; oc++) {
            for (auto k_s = 0; k_s < conv_output_shard_width; k_s++) {
                auto matrix_idx = oc * conv_output_shard_width_padded + k_s;
                auto idx = oc * conv_output_shard_width + k_s;
                output_buffer[matrix_idx] = input_buffer[idx];
            }
        }
        return create_tensor_from_owned_buffer<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };

    return convert_tensor<T>(conv_bias_tensor, compute);
}

// Converts convolution bias to tilized 2d matrix layout for block sharded conv.
// Returns a new tensor with layout=Tile
Tensor convert_conv_bias_tensor_to_tiled_layout_block_sharded(
    const Tensor& conv_bias_tensor, uint32_t num_channel_shards, std::optional<DataType> output_dtype) {
    const static std::unordered_map<
        DataType,
        std::function<Tensor(const Tensor&, uint32_t num_channel_shards, DataType output_dtype)>>
        to_b_tile_layout_map = {
            {DataType::BFLOAT16, &to_bias_tile_layout_block_sharded<bfloat16>},
            {DataType::FLOAT32, &to_bias_tile_layout_block_sharded<float>},
            {DataType::UINT32, &to_bias_tile_layout_block_sharded<uint32_t>},
        };
    return convert_tensor_to_tiled_layout_common(
        conv_bias_tensor, output_dtype, to_b_tile_layout_map, num_channel_shards);
}

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, uint32_t, uint32_t, DataType)>>
        to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_special_padding_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_special_padding_tile_layout<float>},
            {DataType::UINT32, &to_weight_special_padding_tile_layout<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor, output_dtype, to_w_tile_layout_map, in1_block_h, in1_block_w);
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
    auto pad_weight = [&original_weight_shape, &output_weight_shape, &num_groups, &output_dtype](
                          const auto& conv_weight_tensor_buffer) {
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
                    for (int m = 0; m < original_weight_shape[3]; m++) {
                        // Get value from original weight tensor
                        auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<int>{curr_batch_idx, j, k, m}, compute_strides(original_weight_shape));
                        auto value = conv_weight_tensor_buffer[value_flat_input_index];

                        // Copy value to output tensor at the adjusted position
                        auto new_channel_idx = new_channel_start_idx + j;
                        auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<int>{new_batch_idx, new_channel_idx, k, m},
                            compute_strides(output_weight_shape));
                        output_buffer[output_flat_input_index] = value;
                    }
                }
            }
        }
        return Tensor(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_weight_shape, output_dtype, Layout::ROW_MAJOR);
    };

    return convert_tensor<T>(weight, pad_weight);
}

/*
Helper function to aid in converting depthwise weight tensor to broadcasted weight tensor with repeated input channels
*/
template <typename T>
static Tensor conv_depthwise_weight_bcast_helper(
    const Tensor& conv_weight_tensor,
    const ttnn::Shape& original_weight_shape,
    const ttnn::Shape& output_weight_shape,
    DataType output_dtype) {
    auto compute =
        [&original_weight_shape, &output_weight_shape, &output_dtype](const auto& conv_weight_tensor_buffer) {
            ttnn::Shape output_shape = output_weight_shape;
            // Create a new buffer with the output shape
            auto output_buffer = std::vector<T>(output_weight_shape.volume());

            // Copy the original weight tensor to the output tensor
            for (int i = 0; i < output_weight_shape[0]; i++) {
                for (int j = 0; j < output_weight_shape[1]; j++) {
                    for (int k = 0; k < output_weight_shape[2]; k++) {
                        for (int l = 0; l < output_weight_shape[3]; l++) {
                            auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                                ttnn::SmallVector<int>{i, 0, k, l}, compute_strides(original_weight_shape));
                            auto value = conv_weight_tensor_buffer[value_flat_input_index];
                            auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                                ttnn::SmallVector<int>{i, j, k, l}, compute_strides(output_weight_shape));
                            output_buffer[output_flat_input_index] = value;
                        }
                    }
                }
            }
            return Tensor(
                tt::tt_metal::HostBuffer(std::move(output_buffer)),
                output_weight_shape,
                output_dtype,
                Layout::ROW_MAJOR);
        };
    return convert_tensor<T>(conv_weight_tensor, compute);
}

/*
Converts convolution weights to grouped layout with padded zeros
This function will take in a weight tensor with shape [out_channels, in_channels // groups, H, W] and return a newly
allocated output tensor with shape [out_channels, in_channels, H, W] The extra channels in shape[1] will be padded with
0 - then the entire weight tensor is convolved with the input tensor - equivalent to convolution if the input tensor was
divided into num_groups for each groupped filter
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
        original_conv_weight_tensor_shape[3]};

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

    if (tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for Conv2D with groups > 1 expects weights on host, but they are on device. The op will "
            "move them back to host.");
    }
    return convert_tensor_to_tiled_layout_common(
        tt_metal::is_device_tensor(conv_weight_tensor) ? ttnn::operations::core::from_device(conv_weight_tensor)
                                                       : conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape,
        num_groups);
}

/*
Converts convolution weights to depthwise layout
This function will take in a weight tensor with shape [out_channels, 1, H, W] and return a newly
allocated output tensor with shape [out_channels, act_block_h, H, W] The extra channels in shape[1] are repeated
from the original weight tensor - it would be convolving act_block in conv_matrix in one go
*/
Tensor convert_conv_weight_tensor_to_depthwise_layout(
    const Tensor& conv_weight_tensor, uint32_t act_block_h_ntiles, DataType output_dtype) {
    const auto& original_conv_weight_tensor_shape = conv_weight_tensor.logical_shape();
    uint32_t num_input_channels_to_repeat = act_block_h_ntiles * constants::TILE_HEIGHT;
    ttnn::Shape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        num_input_channels_to_repeat,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3]};

    // Create newly allocated buffer all initialized to 0 depending on the datatype of the weight tensor
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::Shape, ttnn::Shape, DataType)>>
        to_w_tile_layout_map = {
            {DataType::INT32, &conv_depthwise_weight_bcast_helper<int32_t>},
            {DataType::FLOAT32, &conv_depthwise_weight_bcast_helper<float>},
            {DataType::BFLOAT16, &conv_depthwise_weight_bcast_helper<bfloat16>},
            {DataType::UINT16, &conv_depthwise_weight_bcast_helper<uint16_t>},
            {DataType::BFLOAT8_B, &conv_depthwise_weight_bcast_helper<float>},
            {DataType::UINT32, &conv_depthwise_weight_bcast_helper<uint32_t>},
            {DataType::BFLOAT4_B, &conv_depthwise_weight_bcast_helper<uint32_t>},
        };
    output_dtype = ((output_dtype == DataType::BFLOAT8_B) || (output_dtype == DataType::BFLOAT4_B)) ? DataType::FLOAT32
                                                                                                    : output_dtype;
    if (tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for Depthwise Conv1D expects weights on host, but they are on device. The op will move "
            "them back to host.");
    }
    return convert_tensor_to_tiled_layout_common(
        tt_metal::is_device_tensor(conv_weight_tensor) ? ttnn::operations::core::from_device(conv_weight_tensor)
                                                       : conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape);
}

static Tensor to_folded_weight_layout(
    const Tensor& conv_weight_tensor,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 4> padding) {
    auto w_shape = conv_weight_tensor.padded_shape();
    uint32_t out_channels = w_shape[0];
    uint32_t in_channels = w_shape[1];
    uint32_t kernel_h = w_shape[2];
    uint32_t kernel_w = w_shape[3];

    // Get input data type
    auto dtype = conv_weight_tensor.dtype();

    ttnn::Shape output_shape = ttnn::Shape({w_shape[0], w_shape[1] * kernel_h * kernel_w, 1, 1});
    auto storage = std::get<tt::tt_metal::HostStorage>(conv_weight_tensor.storage()).buffer;

    auto fold_weights = [&](auto input_buffer) {
        using T = std::decay_t<decltype(input_buffer[0])>;
        std::vector<T> output_buffer(output_shape.volume());

        uint32_t patch_size = kernel_h * kernel_w * in_channels;
        for (auto oc = 0; oc < out_channels; oc++) {
            uint32_t dst_offset = oc * patch_size;
            uint32_t dst_idx = 0;
            for (auto kh = 0; kh < kernel_h; kh++) {
                for (auto kw = 0; kw < kernel_w; kw++) {
                    for (auto ic = 0; ic < in_channels; ic++) {
                        uint32_t src_idx = ((((oc * in_channels + ic) * kernel_h) + kh) * kernel_w) + kw;
                        output_buffer[dst_offset + dst_idx++] = input_buffer[src_idx];
                    }
                }
            }
        }

        return Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), output_shape, dtype, Layout::ROW_MAJOR);
    };

    switch (dtype) {
        case DataType::FLOAT32: return fold_weights(tt::tt_metal::host_buffer::get_as<float>(storage));
        case DataType::BFLOAT16: return fold_weights(tt::tt_metal::host_buffer::get_as<bfloat16>(storage));
        case DataType::UINT32: return fold_weights(tt::tt_metal::host_buffer::get_as<uint32_t>(storage));
        case DataType::INT32: return fold_weights(tt::tt_metal::host_buffer::get_as<int32_t>(storage));
        case DataType::UINT16: return fold_weights(tt::tt_metal::host_buffer::get_as<uint16_t>(storage));
        case DataType::BFLOAT8_B: return fold_weights(tt::tt_metal::host_buffer::get_as<float>(storage));
        case DataType::BFLOAT4_B: return fold_weights(tt::tt_metal::host_buffer::get_as<uint32_t>(storage));
        default:
            TT_THROW(
                "Unsupported input data type for to_folded_weight_layout: {} (type id: {})",
                dtype,
                static_cast<int>(dtype));
    }
}

void validate_weight_tensor(const ttnn::Tensor& weight_tensor) {
    TT_FATAL(weight_tensor.layout() == Layout::ROW_MAJOR, "conv weight layout should be in row_major layout");
    TT_FATAL(weight_tensor.logical_shape().rank() == 4, "conv weight should be 4D tensor");
}

void validate_bias_tensor(const ttnn::Tensor& bias_tensor) {
    TT_FATAL(!ttnn::has_storage_type_of(bias_tensor, ttnn::DEVICE_STORAGE_TYPE), "conv bias should be placed on host");
    TT_FATAL(bias_tensor.logical_shape().rank() == 4, "bias tensor should be 4D tensor");
    TT_FATAL(bias_tensor.layout() == Layout::ROW_MAJOR, "bias tensor layout should be in row_major layout");
}

void validate_weights_format(const std::string& weights_format) {
    TT_FATAL(weights_format.size() == 4, "weights_format must have exactly 4 characters");
    TT_FATAL(weights_format.find("O") != string::npos, "weights_format must contain \"O\"");
    TT_FATAL(weights_format.find("I") != string::npos, "weights_format must contain \"I\"");
    TT_FATAL(weights_format.find("H") != string::npos, "weights_format must contain \"H\"");
    TT_FATAL(weights_format.find("W") != string::npos, "weights_format must contain \"W\"");
    TT_FATAL(weights_format == "OIHW", "Conv2d weights format must be \"OIHW\"");
}

template <typename T>
ttnn::Tensor conv_bias_layout_convert(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    T* device,
    uint32_t out_channels) {
    ttnn::Tensor bias_tensor_ = bias_tensor;
    validate_bias_tensor(bias_tensor_);
    const auto& bias_shape = bias_tensor_.logical_shape();
    TT_FATAL(bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1, "bias shape is not correct");
    ttnn::Shape bias_channels_padded_shape({1, 1, 32, round_up(out_channels, weight_block_w_ntiles * 32)});
    bias_tensor_ =
        ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D{0, 0, 0, 0}, 0);
    bias_tensor_ = ttnn::to_layout(bias_tensor_, Layout::TILE, std::nullopt, std::nullopt, (T*)nullptr);
    if (bias_tensor_.dtype() != bias_dtype) {
        bias_tensor_ = ttnn::to_dtype(bias_tensor_, bias_dtype);
    }
    return bias_tensor_;
}

template <typename T>
static OptimizedConvBlockConfig get_opt_block_config(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t groups,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    T* device,
    Conv2dConfig& conv_config,
    Layout input_layout,
    const DeviceComputeKernelConfig& compute_config,
    const MemoryConfig& input_memory_config,
    const bool has_bias) {
    auto compute_grid_size = device->compute_with_storage_grid_size();

    conv_config = determine_conv_config_for_auto_shard(
        conv_config,
        mm_conv,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        kernel_size[1],
        input_height,
        input_width,
        compute_grid_size,
        input_layout,
        conv_config.dtype,
        input_memory_config,
        kernel_size,
        groups,
        has_bias,
        compute_config);

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    if (input_memory_config.is_sharded() && !conv_config.reshard_if_not_optimal) {
        conv_config.shard_layout = input_memory_config.memory_layout();
    }
    ParallelConfig parallel_config;
    if (input_memory_config.shard_spec().has_value() && !conv_config.reshard_if_not_optimal) {
        parallel_config = {
            .grid = input_memory_config.shard_spec().value().grid,
            .shard_scheme = input_memory_config.memory_layout(),
            .shard_orientation = input_memory_config.shard_spec().value().orientation};
    } else {
        parallel_config = determine_parallel_config(
            conv_config.shard_layout.value(),
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            compute_grid_size,
            shard_orientation,
            !mm_conv,
            true,
            true,
            conv_config.act_block_h_override);
    }
    auto output_parallel_config = parallel_config;
    if (conv_config.shard_layout.value() == ttnn::TensorMemoryLayout::WIDTH_SHARDED && !mm_conv) {
        uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
        output_parallel_config = {
            .grid = tt::tt_metal::num_cores_to_corerangeset(
                find_closest_largest_divisor(tt::div_up(out_channels, tt::constants::TILE_WIDTH), max_num_cores),
                compute_grid_size,
                true),
            .shard_scheme = ttnn::TensorMemoryLayout::WIDTH_SHARDED,
            .shard_orientation = parallel_config.shard_orientation};
        log_debug(tt::LogOp, "Changing width sharded output grid to  {}", output_parallel_config.grid);
    }

    auto conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(
            {1,
             1,
             batch_size * output_height * output_width,
             tt::round_up(
                 out_channels,
                 get_num_cores_channels_from_parallel_config(output_parallel_config) * tt::constants::TILE_WIDTH)}),
        output_parallel_config,
        tt::constants::TILE_HEIGHT);
    auto largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores()
                                       ? output_parallel_config
                                       : parallel_config;
    auto opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config,
        get_num_cores_nhw_from_parallel_config(largest_parallel_config),
        get_num_cores_channels_from_parallel_config(parallel_config));

    const uint32_t in_channels_alignment =
        get_input_channels_alignment(conv_config.shard_layout.value(), input_layout, mm_conv, input_memory_config);
    uint32_t in_channels_padded =
        tt::round_up(in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * in_channels_alignment);

    uint32_t nhw_out_padded_ntile_per_core =
        conv_out_memory_config.shard_spec().value().shape[0] / tt::constants::TILE_HEIGHT;

    return determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile_per_core,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        get_fp32_dest_acc_en(compute_config),
        conv_config.enable_split_reader);
}

template <typename T>
ttnn::Tensor prepare_bias_on_device(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t out_channels,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& input_parallel_config,
    const ParallelConfig& output_parallel_config,
    T* device) {
    uint32_t output_num_cores_channels = get_num_cores_channels_from_parallel_config(output_parallel_config);

    uint32_t out_channels_padded = tt::round_up(out_channels, output_num_cores_channels * tt::constants::TILE_WIDTH);
    uint32_t out_channel_padding = out_channels_padded - out_channels;

    ttnn::Tensor bias_tensor_ = bias_tensor;
    bool is_bias_tensor_is_on_device = tt::tt_metal::is_device_tensor(bias_tensor_);
    if (!is_bias_tensor_is_on_device) {
        bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
    }
    if (input_parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        auto bias_out_channels = bias_tensor_.logical_shape()[3];
        ttnn::Shape bias_channels_padded_shape({1, 1, 1, out_channels_padded});
        bias_tensor_ = ttnn::pad(
            bias_tensor_,
            bias_channels_padded_shape.to_array_4D(),
            tt::tt_metal::Array4D{0, 0, 0, 0},
            0,
            true,
            std::nullopt);
        auto out_channels_per_core = out_channels_padded / output_num_cores_channels;
        auto rounded_weight_block_width = tt::round_up(out_channels_per_core, constants::TILE_WIDTH);

        auto final_out_channels_padded = rounded_weight_block_width * output_num_cores_channels;

        if (final_out_channels_padded != out_channels_padded) {
            bias_tensor_ =
                ttnn::reshape(bias_tensor_, ttnn::Shape({1, 1, output_num_cores_channels, out_channels_per_core}));

            bias_tensor_ = ttnn::pad(
                bias_tensor_,
                tt::tt_metal::Array4D({1, 1, output_num_cores_channels, rounded_weight_block_width}),
                tt::tt_metal::Array4D({0, 0, 0, 0}),
                0,
                true,
                std::nullopt);
        }
        bias_tensor_ = ttnn::reshape(bias_tensor_, ttnn::Shape({1, 1, 1, final_out_channels_padded}));
        bias_tensor_ = ttnn::pad(
            bias_tensor_,
            tt::tt_metal::Array4D({1, 1, 32, final_out_channels_padded}),
            tt::tt_metal::Array4D{0, 0, 0, 0},
            0,
            true,
            std::nullopt);
    } else {
        ttnn::Shape bias_channels_padded_shape({1, 1, 32, round_up(out_channels, weight_block_w_ntiles * 32)});
        bias_tensor_ = ttnn::pad(
            bias_tensor_,
            bias_channels_padded_shape.to_array_4D(),
            tt::tt_metal::Array4D{0, 0, 0, 0},
            0,
            true,
            std::nullopt);
    }
    bias_tensor_ = ttnn::tilize(
        bias_tensor_,
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM),
        bias_dtype,
        true);

    ttnn::Shape bias_target_shape(std::array<uint32_t, 4>{1, 1, 1, out_channels});
    bias_tensor_ = ttnn::reshape(bias_tensor_, bias_target_shape, bias_tensor_.padded_shape());
    return bias_tensor_;
}

template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_on_device(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    T* device) {
    ttnn::Tensor weight_tensor_ = weight_tensor;  // tensor to return
    Shape weight_shape = weight_tensor.logical_shape();
    // In case of 1D convolution and 3D weight tensor, reinterpret it as 4D tensor
    if (weight_shape.rank() == 3 && params.input_width == 1) {
        weight_tensor_ = ttnn::reshape(weight_tensor_, Shape({weight_shape[0], weight_shape[1], weight_shape[2], 1}));
    }

    validate_weight_tensor(weight_tensor_);

    auto original_weights_shape = weight_tensor_.logical_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    ttnn::Tensor bias_tensor_;
    const bool is_conv1d = is_1d_conv(original_weights_window_w, params.input_width);
    const bool is_conv_1d_depthwise_conv = is_1d_deptwise_conv(
        params.groups,
        original_weights_in_channels * params.groups,
        original_weights_out_channels,
        original_weights_window_w,
        params.input_width,
        params.has_bias);

    // Handle kernel stride folding for weights if enabled
    if (params.enable_kernel_stride_folding &&
        (params.stride[0] == original_weights_window_h && params.stride[1] == original_weights_window_w)) {
        // Validate padding is zero for folding
        TT_FATAL(
            params.padding_n4[0] == 0 && params.padding_n4[1] == 0 && params.padding_n4[2] == 0 &&
                params.padding_n4[3] == 0,
            "Padding must be 0 for folding");

        // Move to device if needed
        if (!tt::tt_metal::is_device_tensor(weight_tensor_)) {
            weight_tensor_ = ttnn::to_device(weight_tensor_, device, ttnn::DRAM_MEMORY_CONFIG);
        }

        // Use the fold_tensor utility from conv2d_utils
        weight_tensor_ = fold_tensor(
            weight_tensor_,
            device,
            params.stride,
            {original_weights_window_h, original_weights_window_w},
            params.padding_n4,
            params.weights_bias_dtype,
            true);
    }

    if (!is_conv1d and params.groups > 1) {
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
    } else if (is_conv1d and params.groups > 1) {
        if (is_conv_1d_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(
                weight_tensor_, params.act_block_h_ntiles, weight_tensor_.dtype());
            params.weight_block_h_ntiles = params.act_block_h_ntiles;
        } else {
            weight_tensor_ =
                convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
        }
    }

    weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);

    auto weights_shape = weight_tensor_.logical_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];

    uint32_t input_num_cores_channels = get_num_cores_channels_from_parallel_config(params.input_parallel_config);
    uint32_t output_num_cores_channels = get_num_cores_channels_from_parallel_config(params.output_parallel_config);

    uint32_t out_channels_padded = tt::round_up(out_channels, output_num_cores_channels * tt::constants::TILE_WIDTH);
    uint32_t in_channels_padded = tt::round_up(in_channels, input_num_cores_channels * params.input_channels_alignment);
    uint32_t out_channel_padding = out_channels_padded - out_channels;

    TT_ASSERT(weight_tensor_.layout() == Layout::ROW_MAJOR, "Conv Weights should be in row major layout ");

    // Block sharding re-orders the weights by dividing the input_channels along number of in_channel_cores.
    if (params.input_parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        weight_tensor_ = ttnn::permute(weight_tensor_, ttnn::SmallVector<int64_t>({2, 3, 1, 0}));

        ttnn::Shape weights_channels_padded_shape(
            std::array<uint32_t, 4>({window_h, window_w, out_channels_padded, in_channels_padded}));

        weight_tensor_ = ttnn::pad(
            weight_tensor_,
            tt::tt_metal::Array4D({window_h, window_w, in_channels_padded, out_channels_padded}),
            tt::tt_metal::Array4D({0, 0, 0, 0}),
            0.0f,
            true,
            std::nullopt);

        TT_FATAL(
            input_num_cores_channels == output_num_cores_channels,
            "Input and output cores must be the same for Block Sharded Conv2d");
        TT_FATAL(
            in_channels_padded % input_num_cores_channels == 0,
            "Input channels {} must be divisble by num cores {}",
            in_channels_padded,
            input_num_cores_channels);
        auto in_channels_per_core = in_channels_padded / input_num_cores_channels;

        TT_FATAL(
            out_channels_padded % output_num_cores_channels == 0,
            "output channels {} must be divisble by num cores {}",
            out_channels_padded,
            output_num_cores_channels);
        auto out_channels_per_core = out_channels_padded / output_num_cores_channels;
        auto rounded_weight_block_height =
            tt::round_up(window_h * window_w * in_channels_per_core, constants::TILE_HEIGHT);
        auto rounded_weight_block_width = tt::round_up(out_channels_per_core, constants::TILE_WIDTH);

        auto final_out_channels_padded = rounded_weight_block_width * output_num_cores_channels;

        if (final_out_channels_padded != out_channels_padded) {
            weight_tensor_ = ttnn::reshape(
                weight_tensor_,
                ttnn::Shape(
                    {in_channels_padded * window_h, window_w, output_num_cores_channels, out_channels_per_core}));

            weight_tensor_ = ttnn::pad(
                weight_tensor_,
                tt::tt_metal::Array4D(
                    {in_channels_padded * window_h, window_w, output_num_cores_channels, rounded_weight_block_width}),
                tt::tt_metal::Array4D({0, 0, 0, 0}),
                0,
                true,
                std::nullopt);
        }
        weight_tensor_ = ttnn::reshape(
            weight_tensor_,
            ttnn::Shape(
                {window_h, window_w, input_num_cores_channels, in_channels_per_core, final_out_channels_padded}));

        weight_tensor_ = ttnn::permute(weight_tensor_, ttnn::SmallVector<int64_t>({2, 0, 1, 3, 4}));
        weight_tensor_ = ttnn::reshape(
            weight_tensor_,
            ttnn::Shape(
                {1, input_num_cores_channels, window_h * window_w * in_channels_per_core, final_out_channels_padded}));
        weight_tensor_ = ttnn::pad(
            weight_tensor_,
            tt::tt_metal::Array4D(
                {1, input_num_cores_channels, rounded_weight_block_height, final_out_channels_padded}),
            tt::tt_metal::Array4D({0, 0, 0, 0}),
            0,
            true,
            std::nullopt);

        weight_tensor_ = ttnn::reshape(
            weight_tensor_,
            ttnn::Shape({1, 1, rounded_weight_block_height * input_num_cores_channels, final_out_channels_padded}));
    } else {
        weight_tensor_ = ttnn::permute(weight_tensor_, ttnn::SmallVector<int64_t>({2, 3, 1, 0}));

        ttnn::Shape weights_channels_padded_shape(
            std::array<uint32_t, 4>({window_h, window_w, out_channels_padded, in_channels_padded}));

        weight_tensor_ = ttnn::pad(
            weight_tensor_,
            tt::tt_metal::Array4D({window_h, window_w, in_channels_padded, out_channels_padded}),
            tt::tt_metal::Array4D({0, 0, 0, 0}),
            0.0f,
            true,
            std::nullopt);

        auto weight_block_h_datums = params.weight_block_h_ntiles * constants::TILE_HEIGHT;
        if ((weight_block_h_datums > (window_w * in_channels_padded)) &&
            (params.input_parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED)) {
            weight_tensor_ = ttnn::reshape(
                weight_tensor_, ttnn::Shape({1, window_h, window_w * in_channels_padded, out_channels_padded}));
            weight_tensor_ = ttnn::pad(
                weight_tensor_,
                tt::tt_metal::Array4D({1, window_h, weight_block_h_datums, out_channels_padded}),
                tt::tt_metal::Array4D({0, 0, 0, 0}),
                0.0f,
                true,
                std::nullopt);
            weight_tensor_ = ttnn::reshape(
                weight_tensor_, ttnn::Shape({1, 1, window_h * weight_block_h_datums, out_channels_padded}));
        } else {
            weight_tensor_ = ttnn::reshape(
                weight_tensor_, ttnn::Shape({1, 1, window_h * window_w * in_channels_padded, out_channels_padded}));
        }
    }
    weight_tensor_ = ttnn::tilize(
        weight_tensor_,
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM),
        params.weights_bias_dtype,
        true);

    if (params.weights_bias_dtype.has_value()) {
        TT_ASSERT(
            weight_tensor_.dtype() == params.weights_bias_dtype.value(),
            "Weight tensor should be in the dtype specified by Conv2dConfig");
    }
    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    int32_t weight_matrix_height_padding = weight_tensor_.logical_shape()[2] - weight_matrix_height;
    TT_FATAL(weight_matrix_height_padding >= 0, " Matrix Height Padding can't be negative");

    ttnn::Shape target_shape(std::array<uint32_t, 4>{1, 1, weight_matrix_height, out_channels});

    weight_tensor_ = ttnn::reshape(weight_tensor_, target_shape, weight_tensor_.padded_shape());

    if (bias_tensor.has_value()) {
        bias_tensor_ = prepare_bias_on_device(
            bias_tensor.value(),
            weight_tensor_.dtype(),
            out_channels,
            params.weight_block_w_ntiles,
            params.input_parallel_config,
            params.output_parallel_config,
            device);
        TT_ASSERT(
            bias_tensor_.dtype() == weight_tensor_.dtype(),
            "Bias tensor should be in same dtype as the weights tensor.");
    }
    return {weight_tensor_, bias_tensor.has_value() ? bias_tensor_ : std::optional<ttnn::Tensor>()};
}

template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    T* device) {
    ttnn::Tensor weight_tensor_ = weight_tensor;  // tensor to return
    Shape weight_shape = weight_tensor.logical_shape();
    // In case of 1D convolution and 3D weight tensor, reinterpret it as 4D tensor
    if (weight_shape.rank() == 3 && params.input_width == 1) {
        weight_tensor_ = ttnn::reshape(weight_tensor_, Shape({weight_shape[0], weight_shape[1], weight_shape[2], 1}));
    }
    validate_weight_tensor(weight_tensor_);
    ttnn::Tensor bias_tensor_;

    const auto& original_weights_shape = weight_tensor_.logical_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    const bool is_conv1d = is_1d_conv(original_weights_window_w, params.input_width);
    const bool is_conv_1d_depthwise_conv = is_1d_deptwise_conv(
        params.groups,
        original_weights_in_channels * params.groups,
        original_weights_out_channels,
        original_weights_window_w,
        params.input_width,
        params.has_bias);
    TT_FATAL(
        !is_device_tensor(weight_tensor_),
        "prepare_conv_weights_biases_and_move_to_device is not supported when the weights tensor is on the device");
    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and params.groups > 1) {
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
    } else if (is_conv1d and params.groups > 1) {
        if (is_conv_1d_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(
                weight_tensor_, params.act_block_h_ntiles, weight_tensor_.dtype());
            params.weight_block_h_ntiles = params.act_block_h_ntiles;
        } else {
            weight_tensor_ =
                convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
        }
    }
    if (params.enable_kernel_stride_folding) {
        weight_tensor_ = to_folded_weight_layout(
            weight_tensor_, params.stride, {original_weights_window_h, original_weights_window_w}, params.padding_n4);
    }
    const auto& weights_shape = weight_tensor_.logical_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];

    uint32_t input_num_cores_channels = get_num_cores_channels_from_parallel_config(params.input_parallel_config);
    uint32_t output_num_cores_channels = get_num_cores_channels_from_parallel_config(params.output_parallel_config);

    uint32_t out_channels_padded = tt::round_up(out_channels, output_num_cores_channels * tt::constants::TILE_WIDTH);
    uint32_t in_channels_padded = tt::round_up(in_channels, input_num_cores_channels * params.input_channels_alignment);
    uint32_t out_channel_padding = out_channels_padded - out_channels;

    ttnn::Shape weights_channels_padded_shape({out_channels_padded, in_channels_padded, window_h, window_w});

    weight_tensor_ =
        ttnn::pad(weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);
    // for conv op, pad the weights to block shape
    if (params.input_parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        weight_tensor_ = convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_tensor_, params.weight_block_h_ntiles, params.weight_block_w_ntiles, weight_tensor_.dtype());
    } else if (params.input_parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        weight_tensor_ = convert_conv_weight_tensor_to_tiled_layout_block_sharded(
            weight_tensor_, input_num_cores_channels, weight_tensor_.dtype());
    } else {
        weight_tensor_ = convert_conv_weight_tensor_to_tiled_layout(
            weight_tensor_, params.weight_block_h_ntiles, params.weight_block_w_ntiles, weight_tensor_.dtype());
    }

    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    TT_FATAL(weight_tensor_.logical_shape()[2] >= weight_matrix_height, " Matrix Height Padding can't be negative");
    ttnn::Shape target_shape({1, 1, weight_matrix_height, out_channels});
    ttnn::Shape padded_target_shape({1, 1, weight_tensor_.logical_shape()[2], out_channels + out_channel_padding});
    weight_tensor_ = ttnn::reshape(weight_tensor_, target_shape, padded_target_shape);
    if (params.weights_bias_dtype.has_value()) {
        weight_tensor_ = ttnn::to_dtype(weight_tensor_, params.weights_bias_dtype.value());
    }

    if (params.parameters_on_device) {
        weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);
    }

    if (bias_tensor.has_value()) {
        bias_tensor_ = bias_tensor.value();
        bool is_bias_tensor_is_on_device = tt::tt_metal::is_device_tensor(bias_tensor_);
        if (!is_bias_tensor_is_on_device) {
            TT_FATAL(
                bias_tensor_.logical_shape()[3] == out_channels, "Bias must have the same length as output channels");
            bias_tensor_ = conv_bias_layout_convert(
                bias_tensor_,
                weight_tensor_.dtype(),
                params.weight_block_h_ntiles,
                params.weight_block_w_ntiles,
                params.output_parallel_config,
                device,
                out_channels_padded);
            bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
        }
        TT_ASSERT(
            bias_tensor_.dtype() == weight_tensor_.dtype(),
            "Bias tensor should be in the same dtype as the weights tensor");
    }
    return {weight_tensor_, bias_tensor.has_value() ? bias_tensor_ : std::optional<ttnn::Tensor>()};
}

template <typename T>
ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    const bool has_bias,
    uint32_t groups,
    T* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    if (weights_format != "OIHW") {
        log_warning(
            tt::LogOp,
            "PyTorch expects Conv2D Weights in OIHW format, but got {}. If you have passed the correct weights, then "
            "make sure that the weights_format string is set to \"OIHW\".",
            weights_format);
    }
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    auto orig_stride = stride;
    if (conv_config.enable_kernel_stride_folding) {
        auto folding_result = compute_kernel_stride_folding_params(
            input_height, input_width, in_channels, kernel_size, stride, padding_n4, conv_config);

        input_height = folding_result.input_height;
        input_width = folding_result.input_width;
        in_channels = folding_result.in_channels;
        stride = folding_result.stride;
        kernel_size = folding_result.kernel_size;
        mm_conv = folding_result.mm_conv;
    }
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    if (dram_slice_config_.has_value()) {
        Conv2dSliceConfig dram_slice_config = dram_slice_config_.value();
        const uint32_t output_sliced_dim =
            dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::HEIGHT ? output_height : output_width;
        TT_FATAL(
            dram_slice_config.num_slices > 1, " Number of slices should be greater than 1 for Conv2D DRAM Slicing");
        TT_FATAL(
            dram_slice_config.num_slices < output_sliced_dim,
            " Number of slices should be less than the dimension being sliced in Conv2D DRAM Slicing");

        const uint32_t min_output_slice_size = output_sliced_dim / dram_slice_config.num_slices;
        const uint32_t output_slice_rem = output_sliced_dim % dram_slice_config.num_slices;
        const uint32_t max_output_slice_size = min_output_slice_size + (output_slice_rem > 0);

        if (dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::HEIGHT) {
            output_height = max_output_slice_size;
            input_height =
                ((output_height - 1) * stride[0]) + ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0];
        } else {
            output_width = max_output_slice_size;
            input_width =
                ((output_width - 1) * stride[1]) + ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];
        }
    }
    if (!conv_config.weights_dtype.has_value()) {
        log_warning(
            tt::LogOp,
            "Conv2D prepare_weights was called with conv_config.weights_dtype not set. \n weights_dtype will be set to "
            "the dtype of the input weights tensor. \n Weights & Bias must be the same dtype, so ensure that "
            "conv_weights_dtype is set to the same dtype before calling prepare_bias.");
        conv_config.weights_dtype = weight_tensor.dtype();
    }
    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_height,
        input_width,
        groups,
        kernel_size,
        stride,
        device,
        conv_config,
        input_layout,
        compute_config,
        input_memory_config,
        has_bias);

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    if (input_memory_config.is_sharded() && !conv_config.reshard_if_not_optimal) {
        conv_config.shard_layout = input_memory_config.memory_layout();
    }

    ParallelConfig parallel_config;
    if (input_memory_config.shard_spec().has_value() && !conv_config.reshard_if_not_optimal) {
        parallel_config = {
            .grid = input_memory_config.shard_spec().value().grid,
            .shard_scheme = input_memory_config.memory_layout(),
            .shard_orientation = input_memory_config.shard_spec().value().orientation};
    } else {
        parallel_config = determine_parallel_config(
            conv_config.shard_layout.value(),
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            device->compute_with_storage_grid_size(),
            shard_orientation,
            !mm_conv,
            true,
            true,
            conv_config.act_block_h_override);
    }

    ParallelConfig output_parallel_config = determine_output_parallel_config(
        parallel_config, device->compute_with_storage_grid_size(), out_channels, mm_conv);

    const uint32_t input_channels_alignment =
        get_input_channels_alignment(conv_config.shard_layout.value(), input_layout, mm_conv, input_memory_config);
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt;
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    Conv2dWeightsBiasPrepConfig params(
        input_channels_alignment,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_w_ntiles,
        opt_conv_op_block_config.out_subblock_w_ntiles,
        parallel_config,
        output_parallel_config,
        groups,
        opt_conv_op_block_config.act_block_h_ntiles,
        input_width,
        has_bias,
        true,  // parameters_on_device
        conv_config.enable_kernel_stride_folding,
        kernel_size,
        orig_stride,
        padding_n4);
    if (is_device_tensor(weight_tensor) || conv_config.preprocess_weights_on_device) {
        if (!conv_config.preprocess_weights_on_device) {
            log_warning(
                tt::LogOp,
                "Conv2D prepare weights was invoked with device tensors, but the "
                "conv_config.preprocess_weights_on_device flag was not set to True. \n This will use the device to "
                "prepare weights, which is not fully supported.");
        }
        std::tie(weight_tensor_on_device, bias_tensor_on_device) =
            prepare_conv_weights_biases_on_device(weight_tensor, bias_tensor, params, device);
    } else {
        std::tie(weight_tensor_on_device, bias_tensor_on_device) =
            prepare_conv_weights_biases_and_move_to_device(weight_tensor, bias_tensor, params, device);
    }

    return weight_tensor_on_device;
}

template <typename T>
ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());

    TT_ASSERT(conv_config.weights_dtype.has_value(), "prepare_conv_bias requires conv_config.weights_dtype to be set.");

    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    if (conv_config.enable_kernel_stride_folding) {
        auto folding_result = compute_kernel_stride_folding_params(
            input_height, input_width, in_channels, kernel_size, stride, padding_n4, conv_config);

        input_height = folding_result.input_height;
        input_width = folding_result.input_width;
        in_channels = folding_result.in_channels;
        stride = folding_result.stride;
        kernel_size = folding_result.kernel_size;
        mm_conv = folding_result.mm_conv;
    }
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));

    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_height,
        input_width,
        groups,
        kernel_size,
        stride,
        device,
        conv_config,
        input_layout,
        compute_config,
        input_memory_config,
        true);

    uint32_t weight_block_w_ntiles = opt_conv_op_block_config.out_subblock_w_ntiles;
    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    if (input_memory_config.is_sharded() && !conv_config.reshard_if_not_optimal) {
        conv_config.shard_layout = input_memory_config.memory_layout();
    }
    CoreCoord compute_grid = device->compute_with_storage_grid_size();
    ParallelConfig parallel_config;
    if (input_memory_config.shard_spec().has_value() && !conv_config.reshard_if_not_optimal) {
        parallel_config = {
            .grid = input_memory_config.shard_spec().value().grid,
            .shard_scheme = input_memory_config.memory_layout(),
            .shard_orientation = input_memory_config.shard_spec().value().orientation};
    } else {
        parallel_config = determine_parallel_config(
            conv_config.shard_layout.value(),
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            compute_grid,
            shard_orientation,
            !mm_conv,
            true,
            true,
            conv_config.act_block_h_override);
    }

    ParallelConfig output_parallel_config = determine_output_parallel_config(
        parallel_config, device->compute_with_storage_grid_size(), out_channels, mm_conv);

    ttnn::Tensor bias_tensor_ = bias_tensor;
    TT_FATAL(bias_tensor_.logical_shape()[3] == out_channels, "Bias must have the same length as output channels");

    TT_ASSERT(conv_config.weights_dtype.has_value(), "prepare_conv_bias requires weights_dtype to be set.");
    if (tt::tt_metal::is_device_tensor(bias_tensor_)) {
        bias_tensor_ = prepare_bias_on_device(
            bias_tensor_,
            conv_config.weights_dtype.value(),
            out_channels,
            weight_block_w_ntiles,
            parallel_config,
            output_parallel_config,
            device);
    } else {
        bias_tensor_ = conv_bias_layout_convert(
            bias_tensor_,
            conv_config.weights_dtype.value(),
            opt_conv_op_block_config.act_block_h_ntiles,
            weight_block_w_ntiles,
            output_parallel_config,
            device,
            out_channels);
        bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
    }
    return bias_tensor_;
}

template ttnn::Tensor prepare_conv_weights<IDevice>(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    const bool has_bias,
    uint32_t groups,
    IDevice* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_);

template ttnn::Tensor prepare_conv_weights<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    const bool has_bias,
    uint32_t groups,
    MeshDevice* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_on_device<IDevice>(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    IDevice* device);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_on_device<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    MeshDevice* device);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device<IDevice>(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    IDevice* device);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>>
prepare_conv_weights_biases_and_move_to_device<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    MeshDevice* device);

template ttnn::Tensor prepare_conv_bias<IDevice>(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    IDevice* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template ttnn::Tensor prepare_conv_bias<MeshDevice>(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template ttnn::Tensor conv_bias_layout_convert(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const sliding_window::ParallelConfig& parallel_config,
    IDevice* device,
    uint32_t out_channels);

template ttnn::Tensor conv_bias_layout_convert(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const sliding_window::ParallelConfig& parallel_config,
    MeshDevice* device,
    uint32_t out_channels);

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
