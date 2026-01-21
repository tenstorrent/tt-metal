// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/centralize_w_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace centralize_w_rm {

struct ExecuteCentralizeWRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace centralize_w_rm
}  // namespace operations

// Register the operation
constexpr auto centralize_w_rm =
    ttnn::register_operation<"ttnn::centralize_w_rm", ttnn::operations::centralize_w_rm::ExecuteCentralizeWRm>();

}  // namespace ttnn
