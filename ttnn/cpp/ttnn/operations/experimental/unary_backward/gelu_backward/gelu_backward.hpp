// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::experimental {

Tensor gelu_bw(
    const Tensor& grad_output_tensor,
    const Tensor& input_tensor,
    const std::string& approximate = "none",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> input_grad_tensor = std::nullopt);

}  // namespace ttnn::experimental
