// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"

#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"

#include "device/argmax_op.hpp"

namespace ttnn {
namespace operations::reduction {

struct ExecuteArgMax {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{4, 4, {ttnn::bfloat16}, {ttnn::ROW_MAJOR_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
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

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return execute_on_worker_thread(DefaultQueueId, input_tensor, dim, memory_config, optional_output_tensor);
    }
};

}  // namespace operations::reduction

constexpr auto argmax = ttnn::register_operation<ttnn::operations::reduction::ExecuteArgMax>("ttnn::argmax");

}  // namespace ttnn
