// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::MemoryConfig;

namespace ttnn::experimental {

Tensor gelu_bw(
    const Tensor& grad_output_tensor,
    const Tensor& input_tensor,
    const std::string& approximate,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> input_grad_tensor = std::nullopt);

}  // namespace ttnn::experimental
