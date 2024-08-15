// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/concatenate_heads_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct ConcatenateHeadsOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return operation::run(
                   ConcatenateHeadsDeviceOperation{compute_with_storage_grid_size, memory_config.value_or(input_tensor.memory_config())},
                   {input_tensor}, {}, {optional_output_tensor}, queue_id)
            .at(0);
    }

    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return invoke(
            DefaultQueueId, input_tensor, compute_with_storage_grid_size, memory_config, optional_output_tensor);
    }
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto concatenate_heads = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::concatenate_heads",
    ttnn::operations::experimental::transformer::ConcatenateHeadsOperation>();

}  // namespace experimental

}  // namespace ttnn
