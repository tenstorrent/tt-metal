// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor nlp_concat_heads(
    const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
