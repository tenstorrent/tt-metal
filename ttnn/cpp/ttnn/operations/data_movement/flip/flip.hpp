// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteFlip {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SmallVector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SmallVector<int64_t>& dims);
};

} // namespace operations::data_movement

constexpr auto flip = ttnn::register_operation<"ttnn::flip", ttnn::operations::data_movement::ExecuteFlip>();

} // namespace ttnn
