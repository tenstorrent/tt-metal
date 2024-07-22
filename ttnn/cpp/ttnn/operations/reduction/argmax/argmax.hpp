// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/argmax_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/reduction/argmax/argmax_composite.hpp"

namespace ttnn {
namespace operations::reduction {

struct ExecuteArgMax {
    static ttnn::Tensor operator()(
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

    static ttnn::Tensor operator()(
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return operator()(DefaultQueueId, input_tensor, dim, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        int64_t dim,
        bool all,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return ttnn::operations::unary::_argmax(input_tensor, dim, all, memory_config);
    }
};

}  // namespace operations::reduction

constexpr auto argmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::argmax", ttnn::operations::reduction::ExecuteArgMax>();

}  // namespace ttnn
