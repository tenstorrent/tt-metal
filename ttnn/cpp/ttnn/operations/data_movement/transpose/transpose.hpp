// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement::transpose {

struct ExecuteTranspose {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int64_t dim1,
        int64_t dim2,
        const std::optional<MemoryConfig>& memory_config_arg,
        float pad_value = 0.0f);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, int64_t dim1, int64_t dim2, float pad_value = 0.0f);
};

}  // namespace operations::data_movement::transpose

constexpr auto transpose =
    ttnn::register_operation<"ttnn::transpose", ttnn::operations::data_movement::transpose::ExecuteTranspose>();

}  // namespace ttnn
