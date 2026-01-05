// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_avg_w_rm.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduce_avg_w_rm {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteReduceAvgWRm::invoke(
    const ttnn::Tensor& input,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config) {
    // Call the primitive device operation
    // Unwrap optional memory_config, defaulting to input tensor's memory config
    return ttnn::prim::reduce_avg_w_rm(input, compute_kernel_config, memory_config.value_or(input.memory_config()));
}

}  // namespace ttnn::operations::reduce_avg_w_rm
