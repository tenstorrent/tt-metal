// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

ttnn::Tensor hc_sum_reduce(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt,
    std::optional<MathFidelity> math_fidelity = std::nullopt);

}  // namespace ttnn::experimental
