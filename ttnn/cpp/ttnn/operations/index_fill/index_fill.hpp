// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::index_fill {

struct IndexFill {
    static Tensor invoke(
        const Tensor& input,
        const uint32_t dim,
        const Tensor& index,
        const std::variant<float, int> value,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::index_fill

namespace ttnn {
constexpr auto index_fill =
    ttnn::register_operation_with_auto_launch_op<"ttnn::index_fill", ttnn::operations::index_fill::IndexFill>();
}  // namespace ttnn
