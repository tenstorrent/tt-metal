// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::expand {
struct Expand {
    static Tensor invoke(
        const Tensor& input,
        const std::vector<int32_t>& sizes,

        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::expand

namespace ttnn {
constexpr auto expand =
    ttnn::register_operation_with_auto_launch_op<"ttnn::expand", ttnn::operations::expand::Expand>();
}
