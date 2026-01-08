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
    std::optional<MemoryConfig> output_mem_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    // Call the primitive device operation
    return ttnn::prim::reduce_avg_w_rm(input, output_mem_config, compute_kernel_config);
}

}  // namespace ttnn::operations::reduce_avg_w_rm
