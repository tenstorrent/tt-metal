// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <variant>

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor moreh_full_like(
    const Tensor& input,
    std::variant<float, int> fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
