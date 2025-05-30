// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
namespace ttnn::operations::moreh::moreh_cumsum {
Tensor MorehCumsum::invoke(
    const Tensor& input,
    const int64_t dim,
    std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::experimental::cumsum(DefaultQueueId, input, dim, input.dtype(), output, false, memory_config);
    // return ttnn::prim::moreh_cumsum(input, dim, output, false, memory_config, compute_kernel_config);
}

Tensor MorehCumsumBackward::invoke(
    const Tensor& output_grad,
    const int64_t dim,
    std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::experimental::cumsum(
        DefaultQueueId, output_grad, dim, output_grad.dtype(), input_grad, true, memory_config);

    return ttnn::prim::moreh_cumsum(output_grad, dim, input_grad, true, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
