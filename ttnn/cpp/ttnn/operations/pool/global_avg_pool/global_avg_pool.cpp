// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace tt::tt_metal {

template <PoolType pool>
Tensor pool_2d(
    const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& /*output_dtype*/) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    const auto& input_shape = input.padded_shape();
    switch (pool) {
        case PoolType::AVG: {
            uint32_t height_without_padding = input.logical_shape()[-2];
            return ttnn::operations::reduction::pool_sum(
                input, int(input_shape.rank() - 2), memory_config, std::nullopt, 1 / float(height_without_padding));
        }
        default: TT_THROW("Undefined pool type");
    }
}

Tensor global_avg_pool2d(
    const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");

    // Handle different tensor ranks: 2D [H,W], 3D [H,W,C], or 4D [N,H,W,C]
    auto in_shape = input.padded_shape();
    auto logical_shape = input.logical_shape();
    uint32_t rank = in_shape.rank();

    Tensor input_4d = input;

    // Reshape to 4D if needed
    if (rank == 3) {
        // Rank 3: [H, W, C] -> [1, H, W, C]
        log_debug(
            tt::LogOp,
            "GlobalAvgPool2D: Rank-3 input tensor detected, assuming [H, W, C] format and reshaping to [1, H, W, C]");
        ttnn::Shape reshaped_logical({1, logical_shape[0], logical_shape[1], logical_shape[2]});
        ttnn::Shape reshaped_padded({1, in_shape[0], in_shape[1], in_shape[2]});
        input_4d = ttnn::reshape(input, reshaped_logical, reshaped_padded);
        in_shape = input_4d.padded_shape();
    } else if (rank == 2) {
        // Rank 2: [H, W] -> [1, H, W, 1]
        log_debug(
            tt::LogOp,
            "GlobalAvgPool2D: Rank-2 input tensor detected, assuming [H, W] format and reshaping to [1, H, W, 1]");
        ttnn::Shape reshaped_logical({1, logical_shape[0], logical_shape[1], 1});
        ttnn::Shape reshaped_padded({1, in_shape[0], in_shape[1], 1});
        input_4d = ttnn::reshape(input, reshaped_logical, reshaped_padded);
        in_shape = input_4d.padded_shape();
    } else if (rank != 4) {
        TT_THROW("Input tensor must be rank 2, 3, or 4, got rank {}", rank);
    }

    auto output = input_4d;

    ttnn::Shape output_shape({in_shape[0], 1, in_shape[1] * in_shape[2], in_shape[3]});
    output = ttnn::experimental::view(output, output_shape);

    output = pool_2d<PoolType::AVG>(output, memory_config, output_dtype);
    return output;
}

}  // namespace tt::tt_metal

namespace ttnn {
namespace operations::pool {

Tensor global_avg_pool2d(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DataType>& output_dtype) {
    auto memory_config = memory_config_arg.value_or(input.memory_config());
    auto result = tt::tt_metal::global_avg_pool2d(input, memory_config, output_dtype);
    return result;
}

}  // namespace operations::pool
}  // namespace ttnn
