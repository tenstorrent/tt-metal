// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/variance_w_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace variance_w_rm {

struct ExecuteVarianceWRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace variance_w_rm
}  // namespace operations

// Register the operation
constexpr auto variance_w_rm =
    ttnn::register_operation<"ttnn::variance_w_rm", ttnn::operations::variance_w_rm::ExecuteVarianceWRm>();

}  // namespace ttnn
