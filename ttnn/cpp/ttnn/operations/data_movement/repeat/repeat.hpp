// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

ttnn::Tensor repeat(const ttnn::Tensor& input_tensor, const ttnn::Shape& repeat_dims);

}  // namespace ttnn
