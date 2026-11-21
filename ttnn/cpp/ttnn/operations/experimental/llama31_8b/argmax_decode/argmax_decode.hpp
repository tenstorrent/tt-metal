// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

ttnn::Tensor argmax_decode(
    const ttnn::Tensor& input,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt);

bool argmax_decode_supported(
    const ttnn::Tensor& input,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::experimental::llama31_8b
