// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <variant>

#include <tt-metalium/small_vector.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
enum class Layout;
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::full {
struct Full {
    static ttnn::Tensor invoke(
        const ttnn::SmallVector<uint32_t>& shape,
        const std::variant<float, int> fill_value,
        const ttnn::Tensor& any,
        const std::optional<DataType>& dtype,
        const std::optional<Layout>& layout,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::full

namespace ttnn {
constexpr auto moreh_full =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_full", ttnn::operations::full::Full>();
}  // namespace ttnn
