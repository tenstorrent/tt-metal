// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn {

std::vector<Tensor> sort(
    const Tensor& input_tensor,
    int8_t dim = -1,
    bool descending = false,
    bool stable = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<std::tuple<Tensor&, Tensor&>> optional_output_tensors = std::nullopt);

}  // namespace ttnn
