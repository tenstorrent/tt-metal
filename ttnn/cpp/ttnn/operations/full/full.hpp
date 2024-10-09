// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::full {
struct Full {
    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const float fill_value,
        const ttnn::Tensor any,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const int fill_value,
        const ttnn::Tensor any,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};
}  // namespace ttnn::operations::full

namespace ttnn {
constexpr auto full = ttnn::register_operation_with_auto_launch_op<"ttnn::full", ttnn::operations::full::Full>();
}
