// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/reduce_avg_w_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace reduce_avg_w_rm {

struct ExecuteReduceAvgWRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace reduce_avg_w_rm
}  // namespace operations

// Register the operation
constexpr auto reduce_avg_w_rm =
    ttnn::register_operation<"ttnn::reduce_avg_w_rm", ttnn::operations::reduce_avg_w_rm::ExecuteReduceAvgWRm>();

}  // namespace ttnn
