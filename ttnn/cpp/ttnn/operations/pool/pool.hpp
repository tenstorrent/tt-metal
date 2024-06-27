// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/pool/average_pool.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/cpp/ttnn/operations/pool/max_pool.hpp"
#include "ttnn/cpp/ttnn/operations/conv2d.hpp"

namespace ttnn {
namespace operations {
namespace pool {

namespace detail {
inline const std::array<ttnn::TensorSchema, 1> input_tensor_schemas() {
    return {ttnn::TensorSchema{
        4,  // min rank
        4,  // max rank
        {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::uint16, ttnn::uint32},
        {ttnn::TILE_LAYOUT},
        true,   // can_be_on_device
        false,  // can_be_on_cpu
        false,  // can_be_scalar
        false   // is_optional}
    }};
}
}  // namespace detail

// maxpool macro-op
using array2_t = std::array<uint32_t, 2>;
Tensor maxpool2d(const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, array2_t kernel_size, array2_t stride, array2_t padding, array2_t dilation, Device& device) {
    MemoryConfig memory_config = input_tensor.memory_config();
    const auto shard_grid = memory_config.shard_spec.value().grid;
    const auto shard_scheme = memory_config.memory_layout;
    const auto shard_orientation = memory_config.shard_spec.value().orientation;

    TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
    TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

    ParallelConfig parallel_config = conv2d::determine_parallel_config(
                                        shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                                        batch_size,
                                        0,          // in_channels -- not used
                                        input_h,
                                        input_w,
                                        0,          // out_channels -- not used
                                        device,
                                        shard_orientation);
    uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

    SlidingWindowConfig sliding_window_config = SlidingWindowConfig(batch_size,
                                                                    input_h, input_w,
                                                                    kernel_size.at(0), kernel_size.at(1),
                                                                    stride.at(0), stride.at(1),
                                                                    padding.at(0), padding.at(1),
                                                                    dilation.at(0), dilation.at(1),
                                                                    num_cores_nhw,
                                                                    parallel_config.grid);
    uint32_t neg_inf_pad_val = 0xf7ff;  // TODO: double check

    auto haloed_tensor = ttnn::operations::halo::halo_op(input_tensor, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);
    return tt::tt_metal::maxpool2d_new(haloed_tensor, sliding_window_config, channels, memory_config);
}

struct GlobalAveragePool2D {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<DataType>& output_dtype = std::nullopt) {
        auto memory_config = memory_config_arg.value_or(input.memory_config());
        auto result = tt::tt_metal::average_pool_2d(input, memory_config, output_dtype);
        return result;
    }
};
}  // namespace pool
}  // namespace operations

constexpr auto global_avg_pool2d =
    ttnn::register_operation<ttnn::operations::pool::GlobalAveragePool2D>("ttnn::pool::global_avg_pool2d");

}  // namespace ttnn
