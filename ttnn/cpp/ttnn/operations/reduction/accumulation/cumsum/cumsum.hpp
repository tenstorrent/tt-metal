// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor cumsum(
    const Tensor& input,
    const int32_t& dim,
    std::optional<DataType> dtype = std::nullopt,
    const bool& reverse_order = false,
    std::optional<Tensor> optional_out = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
