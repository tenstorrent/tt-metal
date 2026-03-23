// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

Tensor rotate_half(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
