// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/base_types.hpp>

using tt::tt_metal::DataType;
using tt::tt_metal::MemoryConfig;

namespace ttnn::experimental {

Tensor prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt,
    std::optional<MathFidelity> math_fidelity = std::nullopt);

}  // namespace ttnn::experimental
