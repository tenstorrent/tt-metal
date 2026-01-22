// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/standardize_w_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace standardize_w_rm {

struct ExecuteStandardizeWRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        float epsilon = 1e-5,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace standardize_w_rm
}  // namespace operations

// Register the operation
constexpr auto standardize_w_rm =
    ttnn::register_operation<"ttnn::standardize_w_rm", ttnn::operations::standardize_w_rm::ExecuteStandardizeWRm>();

}  // namespace ttnn
