// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

ttnn::Tensor deepseek_moe_post_combine_tilize(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::experimental
