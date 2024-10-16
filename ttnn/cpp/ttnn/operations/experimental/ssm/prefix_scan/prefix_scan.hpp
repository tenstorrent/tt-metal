// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::ssm {

struct ExecutePrefixScan {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const Tensor& a,
                               const Tensor& bx,
                               const Tensor& h_prev,
                               const std::optional<MemoryConfig>& memory_config = std::nullopt,
                               const std::optional<DataType> dtype = std::nullopt,
                               const std::optional<MathFidelity> math_fidelity = std::nullopt);

    static ttnn::Tensor invoke(const Tensor& a,
                               const Tensor& bx,
                               const Tensor& h_prev,
                               const std::optional<MemoryConfig>& memory_config = std::nullopt,
                               const std::optional<DataType> dtype = std::nullopt,
                               const std::optional<MathFidelity> math_fidelity = std::nullopt);
};

}  // namespace ttnn::operations::experimental::ssm

namespace ttnn::experimental {

constexpr auto prefix_scan =
    ttnn::register_operation_with_auto_launch_op<"ttnn::experimental::prefix_scan",
                                                 ttnn::operations::experimental::ssm::ExecutePrefixScan>();

}  // namespace ttnn::experimental
