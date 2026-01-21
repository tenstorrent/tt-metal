// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/reduce_mean_w_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace reduce_mean_w_rm {

struct ExecuteReduceMeanWRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace reduce_mean_w_rm
}  // namespace operations

// Register the operation
constexpr auto reduce_mean_w_rm =
    ttnn::register_operation<"ttnn::reduce_mean_w_rm", ttnn::operations::reduce_mean_w_rm::ExecuteReduceMeanWRm>();

}  // namespace ttnn
