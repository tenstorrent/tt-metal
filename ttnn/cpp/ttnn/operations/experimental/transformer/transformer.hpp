// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/transformer_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct ConcatenateHeadsOperation {
    static ttnn::Tensor execute_on_worker_thread(
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

    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return execute_on_worker_thread(DefaultQueueId, input_tensor, compute_with_storage_grid_size, memory_config, optional_output_tensor);
    }
};

}  // namespace operations::experimental::transformer

namespace experimental::transformer {

constexpr auto concatenate_heads = ttnn::register_operation<ttnn::operations::experimental::transformer::ConcatenateHeadsOperation>(
    "ttnn::experimental::transformer::concatenate_heads");

}  // namespace experimental::transformer

}  // namespace ttnn
