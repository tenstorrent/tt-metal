// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecutePermute {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor& input_tensor,
                               const std::vector<int64_t>& dims,
                               const std::optional<MemoryConfig>& memory_config,
                               bool composite = true);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor,
                               const std::vector<int64_t>& dims,
                               const std::optional<MemoryConfig>& memory_config);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const std::vector<int64_t>& dims);
};

}  // namespace operations::data_movement

constexpr auto permute = ttnn::register_operation<"ttnn::permute", ttnn::operations::data_movement::ExecutePermute>();

}  // namespace ttnn
