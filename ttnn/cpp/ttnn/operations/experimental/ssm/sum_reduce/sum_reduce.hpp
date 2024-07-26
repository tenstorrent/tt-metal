// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/sum_reduce.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::ssm {

struct ExecuteSumReduce {
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType> dtype = std::nullopt,
        const std::optional<MathFidelity> math_fidelity = std::nullopt) {
        auto program = SumReduce{
            memory_config.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            math_fidelity.value_or(MathFidelity::HiFi4)};
        return operation::run(program, {input_tensor}).at(0);
    }
};

}  // namespace ttnn::operations::experimental::ssm

namespace ttnn {

constexpr auto sum_reduce =
    ttnn::register_operation<"ttnn::experimental::ssm::sum_reduce", operations::experimental::ssm::ExecuteSumReduce>();

}  // namespace ttnn
