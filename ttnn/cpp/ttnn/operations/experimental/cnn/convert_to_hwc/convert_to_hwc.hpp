// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

ttnn::Tensor convert_to_hwc(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DataType>& dtype = std::nullopt);

}  // namespace ttnn::experimental
