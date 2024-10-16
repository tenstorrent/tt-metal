// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

struct ConcatOperation {
    // Wrapper for TTDNN
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const std::vector<ttnn::Tensor>& input_tensors,
                               int dim,
                               const std::optional<MemoryConfig>& memory_config = std::nullopt,
                               std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(const std::vector<ttnn::Tensor>& input_tensors,
                               int dim,
                               const std::optional<MemoryConfig>& memory_config = std::nullopt,
                               std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto concat =
    ttnn::register_operation_with_auto_launch_op<"ttnn::concat", ttnn::operations::data_movement::ConcatOperation>();

}  // namespace ttnn
