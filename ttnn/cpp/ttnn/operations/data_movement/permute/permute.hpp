// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value = 0.0f);

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor, const SmallVector<int64_t>& dims, const std::optional<float>& pad_value = 0.0f);

}  // namespace ttnn
