
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

ttnn::Tensor repeat_and_interleave_eltwise_mul(
    const Tensor& a,
    const Tensor& b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt,
    std::optional<MathFidelity> math_fidelity = std::nullopt);

}  // namespace ttnn::experimental
