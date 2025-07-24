// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

bool is_flip_nop(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& dims) {
    const auto& shape = input_tensor.get_logical_shape();
    for (auto dim : dims) {
        if (shape[dim] > 1) {
            return false;
        }
    }
    return true;  // All flip dimensions have size 1, so it's a no-op
}

ttnn::Tensor flip_impl(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& dims, const MemoryConfig& memory_config) {
    // For tensors with rank < 4, pad to 4D for device operation compatibility
    // const auto rank = input_tensor.get_logical_shape().rank();
    log_debug(tt::LogOp, "flip_impl");
    auto output = ttnn::prim::flip(input_tensor, dims, memory_config, std::nullopt);

    // is_padded is TRUE when input tensor layout is tiled
    // and one or more tensor dims are not divisible by 32
    auto input_shape = input_tensor.logical_shape();
    uint32_t pad_y = (32 - input_shape[2] % 32);
    uint32_t pad_x = (32 - input_shape[3] % 32);
    bool is_padded = (pad_y != 32) || (pad_x != 32);
    is_padded = false;

    log_debug(tt::LogOp, "pad_y: {}", pad_y);
    log_debug(tt::LogOp, "pad_x: {}", pad_x);

    // TODO unpad supports only host tensors TT
    // TODO we should not change the layout
    if (is_padded) {
        output = ttnn::operations::core::from_device(output);
        output = ttnn::to_layout(output, ttnn::Layout::ROW_MAJOR);
        output = output.unpad(
            ttnn::Shape({0, 0, pad_y, pad_x}),
            ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]}));
        // output = output.pad(
        //     ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]}),
        //     ttnn::Shape({0, 0, 0, 0}), 0);
        // output = ttnn::to_layout(output, ttnn::Layout::TILE);
    }
    return output;
}

} // namespace detail

ttnn::Tensor ExecuteFlip::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_rank = input_tensor.logical_shape().rank();

    log_debug(tt::LogOp, "ExecuteFlip::invoke");

    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);
    TT_FATAL(!dims.empty(), "Flip dimensions cannot be empty");
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    // Normalize dimensions to positive indices
    SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [input_tensor](std::int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });

    auto mem_conf = memory_config.value_or(input_tensor.memory_config());

    // Check for no-op case
    log_debug(tt::LogOp, "dims: {}", dims);
    log_debug(tt::LogOp, "normalized_dims: {}", normalized_dims);

    bool is_flip_nop = detail::is_flip_nop(input_tensor, normalized_dims);
    log_debug(tt::LogOp, "is_flip_nop: {}", is_flip_nop);

    if (is_flip_nop) {
        return ttnn::to_memory_config(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    }

    return detail::flip_impl(input_tensor, normalized_dims, mem_conf);
}

ttnn::Tensor ExecuteFlip::invoke(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, dims, memory_config);
}

ttnn::Tensor ExecuteFlip::invoke(const ttnn::Tensor& input_tensor, const SmallVector<int64_t>& dims) {
    return invoke(input_tensor, dims, std::nullopt);
}

} // namespace ttnn::operations::data_movement
