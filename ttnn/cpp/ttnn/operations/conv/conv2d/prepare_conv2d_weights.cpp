// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"

#include "tt_metal/common/work_split.hpp"

#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv2d {

template <typename T, typename compute_>
Tensor convert_tensor(const Tensor& input_tensor, compute_& compute) {
    auto convert_tensor = [&compute](const auto& input_tensor) {
        return std::visit(
            [&compute](auto&& storage) -> Tensor {
                using StorageType = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                    return compute(owned_buffer::get_as<T>(storage.buffer));
                } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                    return compute(borrowed_buffer::get_as<T>(storage.buffer));
                } else {
                    TT_THROW("Unsupported storage type");
                }
            },
            input_tensor.get_storage());
    };

    return ttnn::distributed::is_multi_device_tensor(input_tensor) ? transform(input_tensor, convert_tensor)
                                                                   : convert_tensor(input_tensor);
}

template <typename Func, typename... Args>
Tensor convert_tensor_to_tiled_layout_common(
    const Tensor& input_tensor,
    std::optional<DataType> output_dtype,
    const std::unordered_map<DataType, Func>& function_map,
    Args&&... args) {
    TT_ASSERT(
        input_tensor.get_layout() == Layout::ROW_MAJOR &&
        "Tensor(weight/bias) should be in row major layout for conversion to tilized layout.");

    if (output_dtype.has_value()) {
        if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
            TT_ASSERT(input_tensor.get_dtype() == DataType::FLOAT32);
        } else {
            TT_ASSERT(input_tensor.get_dtype() == input_tensor.get_dtype());
        }
    }
    auto entry = function_map.find(input_tensor.get_dtype());
    if (entry == function_map.end()) {
        TT_THROW("Unsupported data type");
    }
    return entry->second(input_tensor, std::forward<Args>(args)..., output_dtype.value_or(input_tensor.get_dtype()));
}

template <typename T>
Tensor create_tensor_from_owned_buffer(
    owned_buffer::Buffer<T>& buf, DataType& output_dtype, ttnn::SimpleShape& output_shape) {
    if constexpr (std::is_same<T, float>::value) {
        if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
            auto tensor =
                Tensor(std::move(OwnedStorage{std::move(buf)}), output_shape, DataType::FLOAT32, Layout::ROW_MAJOR)
                    .to(Layout::TILE);
            auto output_float_data = owned_buffer::get_as<float>(tensor).get();
            auto output_packed_data =
                output_dtype == DataType::BFLOAT8_B
                    ? pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false)
                    : pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
            return Tensor(
                std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::TILE);
        }
    } else {
        TT_FATAL(
            (output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B),
            "Unsupported output datatype");
    }
    auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(buf)}), output_shape, output_dtype, Layout::ROW_MAJOR);
    return rm_tensor.to(Layout::TILE);
}

template <typename T>
Tensor to_weight_special_padding_tile_layout(
    const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.get_legacy_shape();
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
        ttnn::SimpleShape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(output_shape.volume());
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
        return create_tensor_from_owned_buffer<T>(output_buffer, output_dtype, output_shape);
    };
    return convert_tensor<T>(conv_weight_tensor, compute);
}

template <typename T>
Tensor to_weight_tile_layout(
    const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.get_legacy_shape();
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
        ttnn::SimpleShape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(output_shape.volume());
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
        return create_tensor_from_owned_buffer<T>(output_buffer, output_dtype, output_shape);
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
    auto w_shape = conv_weight_tensor.get_legacy_shape();
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
        ttnn::SimpleShape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(output_shape.volume());
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
        return create_tensor_from_owned_buffer<T>(output_buffer, output_dtype, output_shape);
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
    auto b_shape = conv_bias_tensor.get_legacy_shape();
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
        ttnn::SimpleShape output_shape{1, 1, bias_matrix_rows, bias_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(output_shape.volume());
        for (auto oc = 0; oc < num_channel_shards; oc++) {
            for (auto k_s = 0; k_s < conv_output_shard_width; k_s++) {
                auto matrix_idx = oc * conv_output_shard_width_padded + k_s;
                auto idx = oc * conv_output_shard_width + k_s;
                output_buffer[matrix_idx] = input_buffer[idx];
            }
        }
        return create_tensor_from_owned_buffer<T>(output_buffer, output_dtype, output_shape);
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
    const ttnn::SimpleShape& original_weight_shape,
    const ttnn::SimpleShape& output_weight_shape,
    uint32_t num_groups,
    DataType output_dtype) {
    auto pad_weight = [&original_weight_shape, &output_weight_shape, &num_groups, &output_dtype](
                          const auto& conv_weight_tensor_buffer) {
        owned_buffer::Buffer<T> output_buffer = owned_buffer::create<T>(output_weight_shape.volume());
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
                        auto value_flat_input_index = compute_flat_indices(
                            ttnn::SmallVector<int>{curr_batch_idx, j, k, m}, compute_strides(original_weight_shape));
                        auto value = conv_weight_tensor_buffer[value_flat_input_index];

                        // Copy value to output tensor at the adjusted position
                        auto new_channel_idx = new_channel_start_idx + j;
                        auto output_flat_input_index = compute_flat_indices(
                            ttnn::SmallVector<int>{new_batch_idx, new_channel_idx, k, m},
                            compute_strides(output_weight_shape));
                        output_buffer[output_flat_input_index] = value;
                    }
                }
            }
        }
        return Tensor(
            std::move(OwnedStorage{std::move(output_buffer)}), output_weight_shape, output_dtype, Layout::ROW_MAJOR);
    };

    return convert_tensor<T>(weight, pad_weight);
}

/*
Helper function to aid in converting depthwise weight tensor to broadcasted weight tensor with repeated input channels
*/
template <typename T>
static Tensor conv_depthwise_weight_bcast_helper(
    const Tensor& conv_weight_tensor,
    const ttnn::SimpleShape& original_weight_shape,
    const ttnn::SimpleShape& output_weight_shape,
    DataType output_dtype) {
    owned_buffer::Buffer<T> output_buffer = owned_buffer::create<T>(output_weight_shape.volume());
    auto conv_weight_tensor_buffer = borrowed_buffer::get_as<T>(conv_weight_tensor);
    // Copy the original weight tensor to the output tensor
    for (int i = 0; i < output_weight_shape[0]; i++) {
        for (int j = 0; j < output_weight_shape[1]; j++) {
            for (int k = 0; k < output_weight_shape[2]; k++) {
                for (int l = 0; l < output_weight_shape[3]; l++) {
                    auto value_flat_input_index = compute_flat_indices(
                        ttnn::SmallVector<int>{i, 0, k, l}, compute_strides(original_weight_shape));
                    auto value = conv_weight_tensor_buffer[value_flat_input_index];
                    auto output_flat_input_index =
                        compute_flat_indices(ttnn::SmallVector<int>{i, j, k, l}, compute_strides(output_weight_shape));
                    output_buffer[output_flat_input_index] = value;
                }
            }
        }
    }

    auto output_tensor =
        Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_weight_shape, output_dtype, Layout::ROW_MAJOR);
    return output_tensor;
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
    auto original_conv_weight_tensor_shape_test = conv_weight_tensor.get_shape();
    ttnn::SimpleShape original_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape_test[0],
        original_conv_weight_tensor_shape_test[1],
        original_conv_weight_tensor_shape_test[2],
        original_conv_weight_tensor_shape_test[3]};
    ttnn::SimpleShape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        original_conv_weight_tensor_shape[1] * num_groups,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3]};

    const static std::unordered_map<
        DataType,
        std::function<Tensor(const Tensor&, ttnn::SimpleShape, ttnn::SimpleShape, uint32_t, DataType)>>
        to_w_tile_layout_map = {
            {DataType::INT32, &conv_group_weight_zero_pad_helper<int32_t>},
            {DataType::FLOAT32, &conv_group_weight_zero_pad_helper<float>},
            {DataType::BFLOAT16, &conv_group_weight_zero_pad_helper<bfloat16>},
            {DataType::UINT16, &conv_group_weight_zero_pad_helper<uint16_t>},
            {DataType::BFLOAT8_B, &conv_group_weight_zero_pad_helper<float>},
            {DataType::UINT32, &conv_group_weight_zero_pad_helper<uint32_t>},
            {DataType::BFLOAT4_B, &conv_group_weight_zero_pad_helper<uint32_t>},
        };
    output_dtype = output_dtype == DataType::BFLOAT8_B ? DataType::FLOAT32 : output_dtype;

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor,
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
    auto original_conv_weight_tensor_shape_test = conv_weight_tensor.get_shape();
    uint32_t num_input_channels_to_repeat = act_block_h_ntiles * constants::TILE_HEIGHT;
    ttnn::SimpleShape original_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape_test[0],
        original_conv_weight_tensor_shape_test[1],
        original_conv_weight_tensor_shape_test[2],
        original_conv_weight_tensor_shape_test[3]};
    ttnn::SimpleShape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        num_input_channels_to_repeat,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3]};

    // Create newly allocated buffer all initialized to 0 depending on the datatype of the weight tensor
    const static std::
        unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::SimpleShape, ttnn::SimpleShape, DataType)>>
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

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape);
}

void validate_weight_tensor(const ttnn::Tensor& weight_tensor) {
    TT_FATAL(
        !ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE), "conv weight should be placed on host");
    TT_FATAL(weight_tensor.get_layout() == Layout::ROW_MAJOR, "conv weight layout should be in row_major layout");
    TT_FATAL(weight_tensor.get_shape().rank() == 4, "conv weight should be 4D tensor");
}

void validate_bias_tensor(const ttnn::Tensor& bias_tensor) {
    TT_FATAL(!ttnn::has_storage_type_of(bias_tensor, ttnn::DEVICE_STORAGE_TYPE), "conv bias should be placed on host");
    TT_FATAL(bias_tensor.get_shape().rank() == 4, "bias tensor should be 4D tensor");
    TT_FATAL(bias_tensor.get_layout() == Layout::ROW_MAJOR, "bias tensor layout should be in row_major layout");
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
bool check_non_tile_mul_width(T* device, const Conv2dConfig& conv_config, const uint32_t in_channels) {
    auto num_cores_c = conv_config.transpose_shards ? device->compute_with_storage_grid_size().y
                                                    : device->compute_with_storage_grid_size().x;
    auto elem_size = conv_config.weights_dtype == DataType::BFLOAT8_B ? 1 : 2;
    bool is_non_tile_mul_width =
        (conv_config.shard_layout == TensorMemoryLayout::BLOCK_SHARDED) && conv_config.act_block_h_override == 0 &&
        (conv_config.weights_dtype == DataType::BFLOAT8_B || conv_config.weights_dtype == DataType::BFLOAT16) &&
        conv_config.output_layout == Layout::ROW_MAJOR && ((elem_size * in_channels) % (16 * num_cores_c)) == 0;
    return is_non_tile_mul_width;
}

template <typename T>
ttnn::Tensor conv_bias_layout_convert(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    T* device,
    uint32_t out_channels,
    bool is_non_tile_mul_width) {
    ttnn::Tensor bias_tensor_ = bias_tensor;
    validate_bias_tensor(bias_tensor_);
    if (!is_non_tile_mul_width) {
        auto bias_shape = bias_tensor_.get_shape();
        TT_FATAL(
            bias_shape[3] == out_channels && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1,
            "bias shape is not correct");
        tt::tt_metal::LegacyShape bias_channels_padded_shape = tt::tt_metal::LegacyShape(
            std::array<uint32_t, 4>({1, 1, 32, round_up(out_channels, weight_block_w_ntiles * 32)}));
        bias_tensor_ =
            ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D{0, 0, 0, 0}, 0);
        bias_tensor_ = ttnn::to_layout(bias_tensor_, Layout::TILE, std::nullopt, std::nullopt, (T*)nullptr);
        if (bias_tensor_.get_dtype() != bias_dtype) {
            bias_tensor_ = ttnn::to_dtype(bias_tensor_, bias_dtype);
        }
    } else {
        uint32_t num_cores_channels = get_num_cores_channels_from_parallel_config(parallel_config);
        bias_tensor_ =
            convert_conv_bias_tensor_to_tiled_layout_block_sharded(bias_tensor_, num_cores_channels, bias_dtype);
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
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    T* device,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    const DeviceComputeKernelConfig& compute_config,
    const MemoryConfig& input_memory_config) {
    auto compute_grid_size = device->compute_with_storage_grid_size();

    adjust_conv_op_config_for_auto_shard_if_necessary(
        mm_conv,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        kernel_size[1],
        input_width,
        device->compute_with_storage_grid_size(),
        conv_config,
        input_tensor_layout,
        input_memory_config);

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    bool use_non_tile_height = conv_config.shard_layout.value() == TensorMemoryLayout::HEIGHT_SHARDED &&
                               out_channels <= 256 && conv_config.act_block_h_override == 0 &&
                               (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) &&
                               conv_config.output_layout == Layout::ROW_MAJOR;
    use_non_tile_height = use_non_tile_height && conv_config.input_channels_alignment != 16;

    ParallelConfig parallel_config = determine_parallel_config(
        conv_config.shard_layout.value(),
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        device->compute_with_storage_grid_size(),
        shard_orientation,
        !use_non_tile_height);

    auto output_parallel_config = parallel_config;
    if (conv_config.shard_layout.value() == ttnn::TensorMemoryLayout::WIDTH_SHARDED && !mm_conv) {
        uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
        output_parallel_config = {
            .grid = num_cores_to_corerangeset(
                find_closest_largest_divisor(tt::div_up(out_channels, tt::constants::TILE_WIDTH), max_num_cores),
                compute_grid_size,
                true),
            .shard_scheme = ttnn::TensorMemoryLayout::WIDTH_SHARDED,
            .shard_orientation = parallel_config.shard_orientation};
        log_debug(tt::LogOp, "Changing width sharded output grid to  {}", output_parallel_config.grid);
    }

    uint32_t round_up_size = !use_non_tile_height ? tt::constants::TILE_HEIGHT : 1;
    auto conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(
            std::array<uint32_t, 4>{1, 1, batch_size * output_height * output_width, tt::round_up(out_channels, 32)}),
        output_parallel_config,
        round_up_size);
    auto largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores()
                                       ? output_parallel_config
                                       : parallel_config;
    auto opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config,
        get_num_cores_nhw_from_parallel_config(largest_parallel_config),
        get_num_cores_channels_from_parallel_config(parallel_config));

    uint32_t in_channels_padded = tt::round_up(
        in_channels,
        get_num_cores_channels_from_parallel_config(parallel_config) * conv_config.input_channels_alignment);

    uint32_t nhw_out_padded_ntile = get_num_cores_nhw_from_parallel_config(output_parallel_config) *
                                    conv_out_memory_config.shard_spec.value().shape[0] / tt::constants::TILE_HEIGHT;

    return determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        get_fp32_dest_acc_en(compute_config),
        conv_config.enable_split_reader);
}

template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    T* device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device,
    bool is_non_tile_mul_width) {
    validate_weight_tensor(weight_tensor);
    ttnn::Tensor weight_tensor_;  // tensor to return
    ttnn::Tensor bias_tensor_;

    auto original_weights_shape = weight_tensor.get_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    bool is_conv1d = original_weights_window_w == 1 && input_width == 1;
    bool is_depthwise_conv = groups == original_weights_out_channels && original_weights_in_channels == 1;

    weight_tensor_ = weight_tensor;

    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and groups > 1) {
        weight_tensor_ = convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
    } else if (is_conv1d and groups > 1) {
        if (is_depthwise_conv) {
            weight_tensor_ =
                convert_conv_weight_tensor_to_depthwise_layout(weight_tensor_, act_block_h_ntiles, weights_bias_dtype);
            weight_block_h_ntiles = act_block_h_ntiles;
        } else {
            weight_tensor_ = convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
        }
    }

    auto weights_shape = weight_tensor_.get_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];

    uint32_t num_cores_channels = get_num_cores_channels_from_parallel_config(parallel_config);
    uint32_t out_channels_padded = tt::round_up(out_channels, num_cores_channels * tt::constants::TILE_WIDTH);
    uint32_t in_channels_padded = tt::round_up(in_channels, num_cores_channels * input_channels_alignment);
    uint32_t out_channel_padding = out_channels_padded - out_channels;

    tt::tt_metal::LegacyShape weights_channels_padded_shape = tt::tt_metal::LegacyShape(
        std::array<uint32_t, 4>({out_channels_padded, in_channels_padded, window_h, window_w}));
    if (is_non_tile_mul_width) {
        weights_channels_padded_shape = tt::tt_metal::LegacyShape(std::array<uint32_t, 4>(
            {round_up(out_channels, 32), round_up(in_channels, input_channels_alignment), window_h, window_w}));
        out_channels_padded = tt::round_up(out_channels, 32);
    }
    if (weights_bias_dtype == DataType::BFLOAT8_B) {
        TT_ASSERT(weight_tensor_.get_dtype() == DataType::FLOAT32);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == DataType::FLOAT32);
        }
    } else {
        // TODO: fix the need to check this. We should be able to accept any datatype and convert
        TT_ASSERT(weight_tensor_.get_dtype() == weights_bias_dtype);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == weights_bias_dtype);
        }
    }
    weight_tensor_ =
        ttnn::pad(weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    // for conv op, pad the weights to block shape
    if (parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        weight_tensor_ = convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    } else if (parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        weight_tensor_ = convert_conv_weight_tensor_to_tiled_layout_block_sharded(
            weight_tensor_, num_cores_channels, weights_bias_dtype);
    } else {
        weight_tensor_ = convert_conv_weight_tensor_to_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    }

    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    int32_t weight_matrix_height_padding = weight_tensor_.shape()[2] - weight_matrix_height;
    TT_FATAL(weight_matrix_height_padding >= 0, " Matrix Height Padding can't be negative");

    auto target_shape = ttnn::Shape(
        std::array<uint32_t, 4>{1, 1, weight_matrix_height, out_channels},
        std::array<std::array<uint32_t, 2>, 4>{
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, weight_matrix_height_padding},
            std::array<uint32_t, 2>{0, out_channel_padding}});
    weight_tensor_ = ttnn::reshape(weight_tensor_, target_shape);

    if (parameters_on_device) {
        weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);
    }

    if (bias_tensor.has_value()) {
        bias_tensor_ = bias_tensor.value();
        bool is_bias_tensor_is_on_device = ttnn::is_tensor_on_device_or_multidevice(bias_tensor_);
        if (!is_bias_tensor_is_on_device) {
            bias_tensor_ = conv_bias_layout_convert(
                bias_tensor_,
                weights_bias_dtype,
                weight_block_h_ntiles,
                weight_block_w_ntiles,
                parallel_config,
                device,
                out_channels,
                is_non_tile_mul_width);
            bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
        }
    }

    return {weight_tensor_, bias_tensor.has_value() ? bias_tensor_ : std::optional<ttnn::Tensor>()};
}

template <typename T>
ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_) {
    TT_FATAL(
        !ttnn::is_tensor_on_device_or_multidevice(weight_tensor),
        "Error: weight tensor must be on host for preparation.");
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(
        init_device_compute_kernel_config(device->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false));
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups);
    const uint32_t output_height =
        ((input_height - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;
    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_width,
        kernel_size,
        stride,
        device,
        conv_config,
        input_tensor_layout,
        compute_config,
        input_memory_config);

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    bool use_non_tile_height = conv_config.shard_layout.value() == TensorMemoryLayout::HEIGHT_SHARDED &&
                               out_channels <= 256 && conv_config.act_block_h_override == 0 &&
                               (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) &&
                               conv_config.output_layout == Layout::ROW_MAJOR;
    use_non_tile_height = use_non_tile_height && conv_config.input_channels_alignment != 16;

    ParallelConfig parallel_config = determine_parallel_config(
        conv_config.shard_layout.value(),
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        device->compute_with_storage_grid_size(),
        shard_orientation,
        !use_non_tile_height);

    bool is_non_tile_mul_width = check_non_tile_mul_width(device, conv_config, in_channels);
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt;
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
        weight_tensor,
        bias_tensor,
        conv_config.input_channels_alignment,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_w_ntiles,
        opt_conv_op_block_config.out_subblock_w_ntiles,
        parallel_config,
        device,
        groups,
        opt_conv_op_block_config.act_block_h_ntiles,
        input_width,
        false,
        is_non_tile_mul_width);

    return weight_tensor_on_device;
}

template <typename T>
ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_) {
    TT_FATAL(
        !ttnn::is_tensor_on_device_or_multidevice(bias_tensor), "Error: bias tensor must be on host for preparation.");

    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups);
    const uint32_t output_height =
        ((input_height - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(
        init_device_compute_kernel_config(device->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false));
    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_width,
        kernel_size,
        stride,
        device,
        conv_config,
        input_tensor_layout,
        compute_config,
        input_memory_config);

    uint32_t weight_block_w_ntiles = opt_conv_op_block_config.out_subblock_w_ntiles;
    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    bool use_non_tile_height = conv_config.shard_layout.value() == TensorMemoryLayout::HEIGHT_SHARDED &&
                               out_channels <= 256 && conv_config.act_block_h_override == 0 &&
                               (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) &&
                               conv_config.output_layout == Layout::ROW_MAJOR;
    use_non_tile_height = use_non_tile_height && conv_config.input_channels_alignment != 16;

    ParallelConfig parallel_config = determine_parallel_config(
        conv_config.shard_layout.value(),
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        device->compute_with_storage_grid_size(),
        shard_orientation,
        !use_non_tile_height);

    bool is_non_tile_mul_width = check_non_tile_mul_width(device, conv_config, in_channels);
    ttnn::Tensor bias_tensor_ = bias_tensor;
    bias_tensor_ = conv_bias_layout_convert(
        bias_tensor_,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_h_ntiles,
        weight_block_w_ntiles,
        parallel_config,
        device,
        out_channels,
        is_non_tile_mul_width);
    return bias_tensor_;
}

template OptimizedConvBlockConfig get_opt_block_config<Device>(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    Device* device,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    const DeviceComputeKernelConfig& compute_config,
    const ttnn::MemoryConfig& input_memory_config);

template OptimizedConvBlockConfig get_opt_block_config<MeshDevice>(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    MeshDevice* device,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    const DeviceComputeKernelConfig& compute_config,
    const ttnn::MemoryConfig& input_memory_config);

template ttnn::Tensor prepare_conv_weights<Device>(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Device* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template ttnn::Tensor prepare_conv_weights<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device<Device>(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    Device* device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device,
    bool is_non_tile_mul_width);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>>
prepare_conv_weights_biases_and_move_to_device<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    MeshDevice* device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device,
    bool is_non_tile_mul_width);

template ttnn::Tensor prepare_conv_bias<Device>(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Device* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template ttnn::Tensor prepare_conv_bias<MeshDevice>(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
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
    Device* device,
    uint32_t out_channels,
    bool is_non_tile_mul_width);

template ttnn::Tensor conv_bias_layout_convert(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const sliding_window::ParallelConfig& parallel_config,
    MeshDevice* device,
    uint32_t out_channels,
    bool is_non_tile_mul_width);

template bool check_non_tile_mul_width<Device>(
    Device* device, const Conv2dConfig& conv_config, const uint32_t in_channels);

template bool check_non_tile_mul_width<MeshDevice>(
    MeshDevice* device, const Conv2dConfig& conv_config, const uint32_t in_channels);

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
