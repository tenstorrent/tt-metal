// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/argmax_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

struct ExecuteArgMax {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return operation::run(
                   ArgMax{tt::tt_metal::DataType::UINT32, dim, memory_config.value_or(input_tensor.memory_config())},
                   {input_tensor}, {}, {optional_output_tensor}, queue_id)
            .at(0);
    }

    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return invoke(DefaultQueueId, input_tensor, dim, memory_config, optional_output_tensor);
    }

};

}  // namespace operations::reduction

constexpr auto argmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::argmax", ttnn::operations::reduction::ExecuteArgMax>();

}  // namespace ttnn
