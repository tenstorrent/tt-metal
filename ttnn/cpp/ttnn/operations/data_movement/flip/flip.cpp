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
    //    // For tensors with rank < 4, pad to 4D for device operation compatibility
    //    const auto rank = input_tensor.get_logical_shape().rank();

    // Execute device operation
    auto formatted_input_tensor = input_tensor;

    // uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];
    // // WH and CN should be supported without typecast
    // bool wh = N == 0 && C == 1 && H == 3 && W == 2;
    // bool cn = N == 1 && C == 0 && H == 2 && W == 3;
    // bool cnwh = N == 1 && C == 0 && H == 3 && W == 2;
    // bool bfloat8_supported = wh || cn || cnwh;
    // bool typecast = formatted_input_tensor.dtype() == DataType::BFLOAT8_B and !bfloat8_supported &&
    // !input_tensor.is_sharded(); formatted_input_tensor = typecast ? ttnn::typecast(formatted_input_tensor,
    // DataType::BFLOAT16) : formatted_input_tensor;

    auto output = ttnn::prim::flip(formatted_input_tensor, dims, memory_config, std::nullopt);

    return output;
}

} // namespace detail

ttnn::Tensor ExecuteFlip::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(!dims.empty(), "Flip dimensions cannot be empty");
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    const auto input_rank = input_tensor.logical_shape().rank();
    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);

    // Normalize dimensions to positive indices
    SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [input_tensor](std::int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });

    auto mem_conf = memory_config.value_or(input_tensor.memory_config());

    // Check for no-op case
    if (detail::is_flip_nop(input_tensor, normalized_dims)) {
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
