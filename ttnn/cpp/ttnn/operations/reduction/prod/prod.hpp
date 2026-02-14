// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor prod(
    const Tensor& input,
    std::optional<int64_t> dim = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor prod(
    const Tensor& input,
    const Tensor& output,
    SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
