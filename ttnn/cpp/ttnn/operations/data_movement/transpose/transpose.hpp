// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteTranspose {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor& input_tensor,
                               const int64_t& dim1,
                               const int64_t& dim2,
                               const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor,
                               const int64_t& dim1,
                               const int64_t& dim2,
                               const std::optional<MemoryConfig>& memory_config);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const int64_t& dim1, const int64_t& dim2);
};

}  // namespace operations::data_movement

constexpr auto transpose =
    ttnn::register_operation<"ttnn::transpose", ttnn::operations::data_movement::ExecuteTranspose>();

}  // namespace ttnn
