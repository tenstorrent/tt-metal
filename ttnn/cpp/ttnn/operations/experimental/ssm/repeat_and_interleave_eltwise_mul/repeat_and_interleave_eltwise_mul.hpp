
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::ssm {

struct ExecuteRepeatAndInterleaveEltwiseMul {
    static ttnn::Tensor invoke(
        const Tensor& a,
        const Tensor& b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> dtype = std::nullopt,
        std::optional<MathFidelity> math_fidelity = std::nullopt);
};

}  // namespace ttnn::operations::experimental::ssm
