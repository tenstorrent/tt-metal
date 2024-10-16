// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct NonZeroIndicesOperation {
    static std::vector<ttnn::Tensor> invoke(uint8_t queue_id,
                                            const ttnn::Tensor& input_tensor,
                                            const std::optional<MemoryConfig>& memory_config);

    static std::vector<ttnn::Tensor> invoke(const ttnn::Tensor& input_tensor,
                                            const std::optional<MemoryConfig>& memory_config);
};

}  // namespace operations::data_movement

constexpr auto nonzero =
    ttnn::register_operation<"ttnn::nonzero", ttnn::operations::data_movement::NonZeroIndicesOperation>();

}  // namespace ttnn
