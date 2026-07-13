// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor flip(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config);

ttnn::Tensor flip(const ttnn::Tensor& input_tensor, const SmallVector<int64_t>& dims);

}  // namespace ttnn
