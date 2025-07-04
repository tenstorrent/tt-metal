// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

ttnn::Tensor flip_impl(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config) {

    IDevice* device = input_tensor.device();
    uint32_t rank = input_tensor.logical_shape().rank();

    auto formatted_input_tensor = input_tensor;
    auto output = formatted_input_tensor;
    return output;
}

ttnn::Tensor flip_launch(
    const ttnn::Tensor& a,
    const ttnn::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config) {
    return flip_impl(a, dims, output_mem_config);
}

bool is_flip_nop(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int64_t>& dims) {
    const auto& shape = input_tensor.get_logical_shape();

    for (auto dim : dims) {
        auto normalized_dim = shape.get_normalized_index(dim);
        if (shape[normalized_dim] > 1) {
            return false;
        }
    }

    return true;  // All flip dimensions have size 1, so it's a no-op
}

} // namespace detail

ttnn::Tensor ExecuteFlip::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_rank = input_tensor.logical_shape().rank();
    TT_FATAL(is_device_tensor(input_tensor), "Tensor must already be on device");

    auto output_tensor = input_tensor;

    return output_tensor;
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
