// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {

struct NonzeroOperation {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);
};

}  // namespace operations

constexpr auto nonzero = ttnn::register_operation<"ttnn::nonzero", ttnn::operations::NonzeroOperation>();

}  // namespace ttnn
