// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::ssm {

struct ExecuteHCSumReduce {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType> dtype = std::nullopt,
        const std::optional<MathFidelity> math_fidelity = std::nullopt);
};

}  // namespace ttnn::operations::experimental::ssm

namespace ttnn::experimental {

constexpr auto hc_sum_reduce = ttnn::
    register_operation<"ttnn::experimental::hc_sum_reduce", ttnn::operations::experimental::ssm::ExecuteHCSumReduce>();

}  // namespace ttnn::experimental
