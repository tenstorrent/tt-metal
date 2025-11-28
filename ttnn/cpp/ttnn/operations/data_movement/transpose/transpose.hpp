// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteTranspose {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int64_t dim1,
        int64_t dim2,
        const std::optional<MemoryConfig>& memory_config_arg,
        std::optional<float> pad_value = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, int64_t dim1, int64_t dim2, std::optional<float> pad_value = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto transpose =
    ttnn::register_operation<"ttnn::transpose", ttnn::operations::data_movement::ExecuteTranspose>();

}  // namespace ttnn
