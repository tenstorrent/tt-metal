// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_dot_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<const Tensor>& input_grad = std::nullopt,
    const std::optional<const Tensor>& other_grad = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
