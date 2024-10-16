// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement::clone {
struct Clone {
    static Tensor invoke(const Tensor& input,
                         const std::optional<DataType>& dtype,
                         const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::data_movement::clone

namespace ttnn {
constexpr auto clone =
    ttnn::register_operation_with_auto_launch_op<"ttnn::clone", ttnn::operations::data_movement::clone::Clone>();
}  // namespace ttnn
